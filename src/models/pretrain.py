import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pathlib import Path
import logging

"""Custom code"""
from src.models.transformer_utils import ReZero
from src.models.transformer import Performer, Transformer, MaskedLanguageModel, SOP_Decoder

log = logging.getLogger(__name__)


class TransformerEncoder(pl.LightningModule):
    """Transformer with Masked Language Model and Sentence Order Prediction"""

    def __init__(self, hparams):
        super(TransformerEncoder, self).__init__()
        self.hparams.update(hparams)
        self.last_global_step = 0
        # 1. ENCODER-ENCODER/DECODER
        self.transformer = Performer(self.hparams, decoder=False, with_background=True)

        # 2. DECODER BLOCK
        self.task = self.hparams.training_task
        log.info("Training task: %s" % self.task)
        if "mlm" in self.task:
            # Number of outputs (for logging purposes)
            self.num_outputs = self.hparams.vocab_size

            # 2.1. DECODERS
            self.mlm_decoder = MaskedLanguageModel(
                self.hparams, self.transformer.embedding, act="tanh")
            self.sop_decoder = SOP_Decoder(self.hparams)
            # 2.2. LOSS
            # Weighting for the loss functions
            self.register_buffer("sop_weight", torch.tensor(0.2))
            self.register_buffer("mlm_weight", torch.tensor(0.8))
            self.register_buffer("sop_class_weight",
                                 torch.tensor([1/0.8, 1/0.1, 1/0.1]))
            # Loss functions
            self.sop_loss = nn.CrossEntropyLoss(
                weight=self.sop_class_weight, label_smoothing=0.1)
            self.mlm_loss = nn.CrossEntropyLoss(ignore_index=0)
        else:
            raise NotImplementedError()
        
        # 3. METRICS
        self.init_metrics()

    def forward(self, batch):
        """Forward pass that returns the logits for the masked language model and the sequence order prediction task."""
        # 1. ENCODER INPUT
        predicted = self.transformer(
            x=batch["input_ids"].long(),
            padding_mask=batch["padding_mask"].long()
        )
        # 2. MASKED LANGUAGE MODEL
        mlm_pred = self.mlm_decoder(predicted, batch)
        # 3. SEQUENCE ORDER PREDICTION Task
        # Embedding of the CLS token
        sop_pred = self.sop_decoder(predicted[:, 0])

        return mlm_pred, sop_pred

    def training_step(self, batch, batch_idx):
        """Training Step"""
        # 1. ENCODER-DECODER
        mlm_preds, sop_preds = self(batch)
        # 2. LOSS
        mlm_targs = batch["target_tokens"].long()
        sop_targs = batch["target_sop"].long()
        mlm_loss = self.mlm_loss(mlm_preds.permute(0, 2, 1), target=mlm_targs)
        sop_loss = self.sop_loss(sop_preds, target=sop_targs)

        loss = self.sop_weight * sop_loss + self.mlm_weight * mlm_loss

        self.log("train/loss_mlm", mlm_loss.detach(), on_step=True, on_epoch=True)
        self.log("train/loss_cls", sop_loss.detach(), on_step=True, on_epoch=True)

        ## 3. METRICS
        if (self.global_step + 1) % (self.trainer.log_every_n_steps) == 0:
            self.log_metrics(
                predictions=(mlm_preds.detach(), sop_preds.detach()),
                targets=(mlm_targs.detach(),  sop_targs.detach()),
                loss=loss.detach(),
                stage="train",
                on_step=True,
                on_epoch=True,
            )
        return loss

    def on_train_epoch_start(self, *args):
        """On Epoch Start"""
        self.last_global_step = self.global_step
        seed_everything(self.hparams.seed + self.trainer.current_epoch)

    def on_train_epoch_end(self, *kwargs):
        """On Train Epoch End: Redraw the projection of the Attention-related matrices"""
        if self.hparams.attention_type == "performer":
            self.transformer.redraw_projection_matrix(-1)
        else:
            raise NotImplementedError(
                "We only have a Performer implementation.")

    def validation_step(self, batch, batch_idx):
        """Validation Step"""
        # 1. ENCODER-DECODER
        mlm_preds, sop_preds = self(batch)
        # 2. LOSS
        mlm_targs = batch["target_tokens"].long()
        sop_targs = batch["target_sop"].long()
        mlm_loss = self.mlm_loss(mlm_preds.permute(0, 2, 1), target=mlm_targs)
        sop_loss = self.sop_loss(sop_preds, target=sop_targs)

        loss = self.sop_weight * sop_loss + self.mlm_weight * mlm_loss

        self.log("val/loss_mlm", mlm_loss.detach(), on_step=True, on_epoch=True)
        self.log("val/loss_cls", sop_loss.detach(), on_step=True, on_epoch=True)

        ## 3. METRICS
        self.log_metrics(
            predictions=(mlm_preds.detach(), sop_preds.detach()),
            targets=(mlm_targs.detach(),  sop_targs.detach()),
            loss=loss.detach(),
            stage="val",
            on_step=False,
            on_epoch=True,
        )
        return loss

    def test_step(self, batch, batch_idx):
        # 1. ENCODER-DECODER
        mlm_preds, sop_preds = self(batch)
        # 2. LOSS
        mlm_targs = batch["target_tokens"].long()
        sop_targs = batch["target_sop"].long()
        mlm_loss = self.mlm_loss(mlm_preds.permute(0, 2, 1), target=mlm_targs)
        sop_loss = self.sop_loss(sop_preds, target=sop_targs)

        loss = self.sop_weight * sop_loss + self.mlm_weight * mlm_loss

        self.log("test/loss_mlm", mlm_loss.detach(), on_step=False, on_epoch=True)
        self.log("test/loss_cls", sop_loss.detach(), on_step=False, on_epoch=True)

        ## 3. METRICS
        self.log_metrics(
            predictions=(mlm_preds.detach(), sop_preds.detach()),
            targets=(mlm_targs.detach(),  sop_targs.detach()),
            loss=loss.detach(),
            stage="test",
            on_step=False,
            on_epoch=True,
        )
        return loss

    def configure_optimizers(self):
        """Configuration of the Optimizer and the Learning Rate Scheduler."""
        no_decay = [
            "bias",
            "norm",
            "age",
            "year",
            "token",
            "decoder.g"
        ]

        # It is advised to avoid the decay on the embedding weights, biases of the model and values of the ReZero gates.

        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            betas=(self.hparams.beta1, self.hparams.beta2),
            eps=self.hparams.epsilon,
        )

        eff_batch_size = self.hparams.batch_size * self.hparams.accumulate_grad_batches

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                    optimizer, max_lr=self.hparams.learning_rate,
                    epochs=self.hparams.num_epochs, steps_per_epoch=self.hparams.n_users // eff_batch_size + (self.hparams.n_users % eff_batch_size > 0),
                    three_phase=False, pct_start=self.hparams.pct_start, max_momentum=self.hparams.beta1,
                    div_factor=30
                ),
                "interval": "step",
                "frequency": 1,
                "name": "learning_rate",
            },
        }
    
    def init_metrics(self):
        ##### TRAIN
        top_k = 5 if self.num_outputs == self.hparams.vocab_size else 1

        self.train_accuracy = torchmetrics.Accuracy(
            task="multiclass",
            threshold=0.2,
            num_classes=self.num_outputs,
            average="macro",
            multidim_average="global",
            ignore_index=0,
            top_k=top_k,
        )

        self.train_precision = torchmetrics.Precision(
            task="multiclass",
            threshold=0.2,
            num_classes=self.num_outputs,
            average="macro",
            multidim_average="global",
            ignore_index=0,
            top_k=top_k,
        )

        self.train_recall = torchmetrics.Recall(
            task="multiclass",
            threshold=0.2,
            num_classes=self.num_outputs,
            average="macro",
            multidim_average="global",
            ignore_index=0,
            top_k=top_k,
        )

        self.train_f1 = torchmetrics.F1Score(
            task="multiclass",
            threshold=0.2,
            num_classes=self.num_outputs,
            average="macro",
            multidim_average="global",
            ignore_index=0,
            top_k=top_k,
        )

        self.train_cls_acc = torchmetrics.Accuracy(
            task="multiclass",
            threshold=0.5,
            num_classes=3,
            average="macro"
        )

        self.train_cls_f1 = torchmetrics.F1Score(
            task="multiclass",
            threshold=0.5,
            num_classes=3,
            average="macro",
        )

        ##### VALIDATION
        self.val_accuracy = torchmetrics.Accuracy(
            task="multiclass",
            threshold=0.2,
            num_classes=self.num_outputs,
            average="macro",
            multidim_average="global",
            ignore_index=0,
            top_k=top_k,
        )

        self.val_precision = torchmetrics.Precision(
            task="multiclass",
            threshold=0.2,
            num_classes=self.num_outputs,
            average="macro",
            multidim_average="global",
            ignore_index=0,
            top_k=top_k,
        )

        self.val_recall = torchmetrics.Recall(
            task="multiclass",
            threshold=0.2,
            num_classes=self.num_outputs,
            average="macro",
            multidim_average="global",
            ignore_index=0,
            top_k=top_k,
        )

        self.val_f1 = torchmetrics.F1Score(
            task="multiclass",
            threshold=0.2,
            num_classes=self.num_outputs,
            average="macro",
            multidim_average="global",
            ignore_index=0,
            top_k=top_k,
        )

        self.val_cls_acc = torchmetrics.Accuracy(
            task="multiclass",
            threshold=0.5,
            num_classes=3,
            average="macro"
        )

        self.val_cls_f1 = torchmetrics.F1Score(
            task="multiclass",
            threshold=0.5,
            num_classes=3,
            average="macro",
        )

    def log_metrics(
        self,
        predictions,
        targets,
        loss,
        stage,
        on_step: bool = True,
        on_epoch: bool = True,
    ):
        """Compute on step/epoch metrics"""
        loss = loss.cpu()
        cls_preds = F.softmax(predictions[1], dim=-1)
        mlm_preds = F.softmax(predictions[0], dim=-1).permute(0, 2, 1)

        cls_targs = targets[1]
        mlm_targs = targets[0]

        if stage == "train":

            self.log("train/loss", loss, on_step=on_step, on_epoch=on_epoch)
            self.log(
                "train/perplexity", torch.sqrt(loss), on_step=on_step, on_epoch=on_epoch
            )
            self.log(
                "train/accuracy",
                self.train_accuracy(mlm_preds, mlm_targs),
                on_step=on_step,
                on_epoch=on_epoch,
            )
            self.log(
                "train/recall",
                self.train_recall(mlm_preds, mlm_targs),
                on_step=on_step,
                on_epoch=on_epoch,
            )
            self.log(
                "train/precision",
                self.train_precision(mlm_preds, mlm_targs),
                on_step=on_step,
                on_epoch=on_epoch,
            )
            self.log(
                "train/f1",
                self.train_f1(mlm_preds, mlm_targs),
                on_step=on_step,
                on_epoch=on_epoch,
            )
            self.log(
                "train/cls_acc",
                self.train_cls_acc(cls_preds, cls_targs),
                on_step=on_step,
                on_epoch=on_epoch,
            )
            self.log(
                "train/cls_f1",
                self.train_cls_f1(cls_preds, cls_targs),
                on_step=on_step,
                on_epoch=on_epoch,
            )


        elif stage == "val":
            self.log("val/loss", loss, on_step=on_step, on_epoch=on_epoch)
            self.log(
                "val/perplexity", torch.sqrt(loss), on_step=on_step, on_epoch=on_epoch
            )
            self.log(
                "val/accuracy",
                self.val_accuracy(mlm_preds, mlm_targs),
                on_step=on_step,
                on_epoch=on_epoch,
            )
            self.log(
                "val/recall",
                self.val_recall(mlm_preds, mlm_targs),
                on_step=on_step,
                on_epoch=on_epoch,
            )
            self.log(
                "val/precision",
                self.val_precision(mlm_preds, mlm_targs),
                on_step=on_step,
                on_epoch=on_epoch,
            )
            self.log(
                "val/f1",
                self.val_f1(mlm_preds, mlm_targs),
                on_step=on_step,
                on_epoch=on_epoch,
            )
            self.log(
                "val/cls_acc",
                self.val_cls_acc(cls_preds, cls_targs),
                on_step=on_step,
                on_epoch=on_epoch,
            )
            self.log(
                "val/cls_f1",
                self.val_cls_f1(cls_preds, cls_targs),
                on_step=on_step,
                on_epoch=on_epoch,
            )
        
        elif stage == "test":
            self.log("test/loss", loss, on_step=on_step, on_epoch=on_epoch)
            self.log(
                "test/perplexity", torch.sqrt(loss), on_step=on_step, on_epoch=on_epoch
            )
            self.log(
                "test/accuracy",
                self.val_accuracy(mlm_preds, mlm_targs),
                on_step=on_step,
                on_epoch=on_epoch,
            )
            self.log(
                "test/recall",
                self.val_recall(mlm_preds, mlm_targs),
                on_step=on_step,
                on_epoch=on_epoch,
            )
            self.log(
                "test/precision",
                self.val_precision(mlm_preds, mlm_targs),
                on_step=on_step,
                on_epoch=on_epoch,
            )
            self.log(
                "test/f1",
                self.val_f1(mlm_preds, mlm_targs),
                on_step=on_step,
                on_epoch=on_epoch,
            )
            self.log(
                "test/cls_acc",
                self.val_cls_acc(cls_preds, cls_targs),
                on_step=on_step,
                on_epoch=on_epoch,
            )
            self.log(
                "test/cls_f1",
                self.val_cls_f1(cls_preds, cls_targs),
                on_step=on_step,
                on_epoch=on_epoch,
            )
