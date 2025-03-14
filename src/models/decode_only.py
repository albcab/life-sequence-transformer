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
from src.models.transformer import Performer, NextTokenDecoder

log = logging.getLogger(__name__)


class TransformerDecoderOnly(pl.LightningModule):

    def __init__(self, hparams):
        super(TransformerDecoderOnly, self).__init__()
        self.hparams.update(hparams)
        self.last_global_step = 0
        # 1. DECODER
        self.transformer = Performer(self.hparams, decoder=True, with_background=True)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)
        self.num_outputs = self.hparams.vocab_size

        # 2. META PARAM + LAST LAYER
        self.task = self.hparams.training_task
        log.info("Training task: %s" % self.task)
        self.decoder = NextTokenDecoder(self.hparams, self.transformer.embedding, with_background=True)
        
        # 3. METRICS
        self.init_metrics()

    def forward(self, batch):
        # 1. ENCODER INPUT
        predicted = self.transformer(
            x=batch["input_ids"].long(),
            padding_mask=batch["padding_mask"].long(),
        )
        nxt_pred = self.decoder(predicted)

        return nxt_pred

    def training_step(self, batch, batch_idx):
        """Training Step"""
        # 1. ENCODER-DECODER
        predicts = self(batch)
        # 2. LOSS
        targets = batch["target_tokens"].long()
        loss = self.loss_fn(predicts.permute(0, 2, 1), target=targets)

        # self.log("train/loss", loss.detach(), on_step=True, on_epoch=True)

        ## 3. METRICS
        if (self.global_step + 1) % (self.trainer.log_every_n_steps) == 0:
            self.log_metrics(
                predictions=predicts.detach(),
                targets=targets.detach(),
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
        predicts = self(batch)
        # 2. LOSS
        targets = batch["target_tokens"].long()
        loss = self.loss_fn(predicts.permute(0, 2, 1), target=targets)

        self.log("val/loss_step", loss.detach(), on_step=True, on_epoch=True)

        ## 3. METRICS
        self.log_metrics(
            predictions=predicts.detach(),
            targets=targets.detach(),
            loss=loss.detach(),
            stage="val",
            on_step=False,
            on_epoch=True,
        )
        return loss

    def test_step(self, batch, batch_idx):
        # 1. ENCODER-DECODER
        predicts = self(batch)
        # 2. LOSS
        targets = batch["target_tokens"].long()
        loss = self.loss_fn(predicts.permute(0, 2, 1), target=targets)

        self.log("test/loss", loss.detach(), on_step=False, on_epoch=True)

        ## 3. METRICS
        self.log_metrics(
            predictions=predicts.detach(),
            targets=targets.detach(),
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
            "abspos",
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
                    three_phase=False, pct_start=0.05, max_momentum=self.hparams.beta1,
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
        predictions = F.softmax(predictions, dim=-1).permute(0, 2, 1)

        if stage == "train":

            self.log("train/loss", loss, on_step=on_step, on_epoch=on_epoch)
            self.log(
                "train/perplexity", torch.sqrt(loss), on_step=on_step, on_epoch=on_epoch
            )
            self.log(
                "train/accuracy",
                self.train_accuracy(predictions, targets),
                on_step=on_step,
                on_epoch=on_epoch,
            )
            self.log(
                "train/recall",
                self.train_recall(predictions, targets),
                on_step=on_step,
                on_epoch=on_epoch,
            )
            self.log(
                "train/precision",
                self.train_precision(predictions, targets),
                on_step=on_step,
                on_epoch=on_epoch,
            )
            self.log(
                "train/f1",
                self.train_f1(predictions, targets),
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
                self.val_accuracy(predictions, targets),
                on_step=on_step,
                on_epoch=on_epoch,
            )
            self.log(
                "val/recall",
                self.val_recall(predictions, targets),
                on_step=on_step,
                on_epoch=on_epoch,
            )
            self.log(
                "val/precision",
                self.val_precision(predictions, targets),
                on_step=on_step,
                on_epoch=on_epoch,
            )
            self.log(
                "val/f1",
                self.val_f1(predictions, targets),
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
                self.val_accuracy(predictions, targets),
                on_step=on_step,
                on_epoch=on_epoch,
            )
            self.log(
                "test/recall",
                self.val_recall(predictions, targets),
                on_step=on_step,
                on_epoch=on_epoch,
            )
            self.log(
                "test/precision",
                self.val_precision(predictions, targets),
                on_step=on_step,
                on_epoch=on_epoch,
            )
            self.log(
                "test/f1",
                self.val_f1(predictions, targets),
                on_step=on_step,
                on_epoch=on_epoch,
            )
