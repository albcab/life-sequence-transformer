import logging

import pytorch_lightning as pl
import torch
import torchmetrics

from pytorch_lightning import seed_everything
from transformers import (
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
)


log = logging.getLogger(__name__)


class HFCausalLM(pl.LightningModule):

    def __init__(self, hparams, tokenizer):
        super().__init__()
        self.hparams.update(hparams)
        self.last_global_step = 0

        # 1. MODEL
        self.model = AutoModelForCausalLM.from_pretrained(
            self.hparams.model_name,
        )

        # The datamodule added [BOL], [EOL], [EOY], [PLCH0].
        # The HF model therefore needs corresponding embedding rows.
        self.model.resize_token_embeddings(len(tokenizer))
        self.model.config.pad_token_id = tokenizer.pad_token_id

        self.num_outputs = len(tokenizer)

        # 2. META
        self.task = self.hparams.training_task
        log.info("Training task: %s", self.task)
        log.info("HuggingFace model: %s", self.hparams.model_name)
        log.info("Vocabulary size: %s", self.num_outputs)

        # 3. METRICS
        self.init_metrics()

    def forward(self, batch):
        return self.model(
            input_ids=batch["input_ids"].long(),
            attention_mask=batch["attention_mask"].long(),
            labels=batch["labels"].long(),
        )

    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = outputs.loss

        if (self.global_step + 1) % self.trainer.log_every_n_steps == 0:
            self.log_metrics(
                predictions=outputs.logits.detach(),
                targets=batch["labels"],
                loss=loss.detach(),
                stage="train",
                on_step=True,
                on_epoch=True,
            )

        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = outputs.loss

        self.log_metrics(
            predictions=outputs.logits.detach(),
            targets=batch["labels"],
            loss=loss.detach(),
            stage="val",
            on_step=False,
            on_epoch=True,
        )

        return loss

    def test_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = outputs.loss

        self.log_metrics(
            predictions=outputs.logits.detach(),
            targets=batch["labels"],
            loss=loss.detach(),
            stage="test",
            on_step=False,
            on_epoch=True,
        )

        return loss

    def on_train_epoch_start(self):
        self.last_global_step = self.global_step
        seed_everything(
            self.hparams.seed + self.trainer.current_epoch,
            workers=True,
        )

    def configure_optimizers(self):
        """
        AdamW with weight decay on matrix parameters and no weight decay
        on biases/norm parameters, followed by warmup + cosine decay.
        """

        decay_parameters = []
        no_decay_parameters = []

        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue

            if parameter.ndim >= 2:
                decay_parameters.append(parameter)
            else:
                no_decay_parameters.append(parameter)

        optimizer_grouped_parameters = [
            {
                "params": decay_parameters,
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": no_decay_parameters,
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            betas=(self.hparams.beta1, self.hparams.beta2),
            eps=self.hparams.epsilon,
        )

        total_steps = self.trainer.estimated_stepping_batches

        warmup_steps = int(
            total_steps * self.hparams.warmup_ratio
        )

        log.info("Total optimizer steps: %s", total_steps)
        log.info("Warmup steps: %s", warmup_steps)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "name": "learning_rate",
            },
        }

    def init_metrics(self):
        self.accuracy = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=self.num_outputs,
            multidim_average="global",
            ignore_index=-100,
            top_k=5,
        )

        self.accuracy1 = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=self.num_outputs,
            multidim_average="global",
            ignore_index=-100,
            top_k=1,
        )

    def log_metrics(
        self,
        predictions,
        targets,
        loss,
        stage,
        on_step=True,
        on_epoch=True,
    ):
        """
        Log causal-language-model metrics.

        HuggingFace CausalLM models shift logits/labels internally when
        computing loss. We reproduce that shift for token accuracy.
        """

        shift_predictions = predictions[:, :-1, :]
        shift_targets = targets[:, 1:]

        # perplexity = torch.exp(loss)
        perplexity = torch.sqrt(loss)    

        self.log(
            f"{stage}/loss",
            loss,
            on_step=on_step,
            on_epoch=on_epoch,
            prog_bar=stage != "train",
            sync_dist=True,
        )

        self.log(
            f"{stage}/perplexity",
            perplexity,
            on_step=on_step,
            on_epoch=on_epoch,
            sync_dist=True,
        )

        self.log(
            f"{stage}/accuracy",
            self.accuracy(shift_predictions, shift_targets),
            on_step=on_step,
            on_epoch=on_epoch,
            sync_dist=True,
        )

        self.log(
            f"{stage}/accuracy1",
            self.accuracy1(shift_predictions, shift_targets),
            on_step=on_step,
            on_epoch=on_epoch,
            sync_dist=True,
        )