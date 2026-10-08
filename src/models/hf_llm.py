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
        # Accumulate token correctness without vocabulary-sized one-hot tensors.
        self.accuracy = torchmetrics.MeanMetric()
        self.accuracy1 = torchmetrics.MeanMetric()

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

        valid = shift_targets != -100
        top_tokens = shift_predictions.topk(min(5, shift_predictions.shape[-1]), dim=-1).indices
        self.accuracy((top_tokens == shift_targets.unsqueeze(-1)).any(dim=-1)[valid].float())
        self.accuracy1((top_tokens[..., 0] == shift_targets)[valid].float())

        self.log(
            f"{stage}/accuracy",
            self.accuracy,
            on_step=on_step,
            on_epoch=on_epoch,
            sync_dist=True,
        )

        self.log(
            f"{stage}/accuracy1",
            self.accuracy1,
            on_step=on_step,
            on_epoch=on_epoch,
            sync_dist=True,
        )

    @torch.inference_mode()
    def smc_sample(
        self,
        input_ids,
        attention_mask,
        automata,
        tokenizer,
        lexer,
        num_years,
        eoy_idx,
        valid_token_ids,
        nb_particles=32,
        max_new_tokens=50,
        ess_threshold=0.5,
        max_stack_height=12,
        max_steps=100000,
        verbose=False,
    ):
        """
        SMC sampling from the LLM conditioned on the automaton.

        The proposal is the LLM distribution restricted to automaton-valid
        tokens. The incremental importance weight is the probability mass of
        the valid tokens under the original LLM distribution.

        Generation stops for a particle once its generated EOY count is
        greater than num_years.
        """

        device = input_ids.device
        vocab_size = self.num_outputs
        eos_token_id = tokenizer.eos_token_id
        bos = tokenizer.bos_token_id is not None

        if input_ids.shape[0] != 1:
            raise ValueError("smc_sample expects exactly one prompt")
        if nb_particles < 1:
            raise ValueError("nb_particles must be positive")
        if not 0 < ess_threshold <= 1:
            raise ValueError("ess_threshold must be in (0, 1]")

        prompt_len = input_ids.shape[1]

        input_ids = input_ids.repeat(nb_particles, 1)
        attention_mask = attention_mask.repeat(nb_particles, 1)

        done = torch.zeros(nb_particles, dtype=torch.bool, device=device)
        year_counters = torch.zeros(nb_particles, dtype=torch.long, device=device)
        year_limits = torch.full(
            (nb_particles,), int(num_years), dtype=torch.long, device=device
        )
        log_weights = torch.zeros(nb_particles, dtype=torch.float64, device=device)
        trajectory_log_weights = torch.zeros(
            nb_particles, dtype=torch.float64, device=device
        )

        for step in range(max_new_tokens):
            if done.all():
                break

            active = torch.where(~done)[0]

            outputs = self.model(
                input_ids=input_ids[active],
                attention_mask=attention_mask[active],
            )
            logits = outputs.logits[:, -1, :]
            model_log_probs = torch.log_softmax(logits, dim=-1)

            remaining_tokens = max_new_tokens - step - 1
            sampled_tokens = {}

            for row, particle_tensor in enumerate(active):
                particle = int(particle_tensor.item())

                generated_ids = input_ids[particle, prompt_len:]
                generated_mask = attention_mask[particle, prompt_len:].bool()
                generated_ids = generated_ids[generated_mask]

                current_str = "".join(
                    tokenizer.decode([token_id], skip_special_tokens=False)
                    for token_id in generated_ids.tolist()
                )

                valid_tokens = []

                for token_id in valid_token_ids:
                    decoded = tokenizer.decode(
                        [token_id],
                        skip_special_tokens=False,
                    )

                    config, distance = lexer.get_configurations(
                        t=current_str + decoded,
                        q=automata.initial_state,
                        gamma=[automata.stack_bottom],
                        s_pref="",
                        max_stack_height=max_stack_height,
                        max_steps=max_steps,
                        return_first=True,
                        remaining_tokens=remaining_tokens,
                        bos=bos,
                        search_strategy="dfs",
                    )

                    if token_id == eos_token_id and distance != 0:
                        continue

                    if config != ():
                        valid_tokens.append(token_id)

                if not valid_tokens:
                    raise RuntimeError(
                        "Automaton found no valid continuation for "
                        f"particle {particle} at step {step}"
                    )

                valid_tokens = torch.tensor(
                    valid_tokens,
                    dtype=torch.long,
                    device=device,
                )
                valid_model_log_probs = model_log_probs[row, valid_tokens]

                incremental_log_weight = torch.logsumexp(
                    valid_model_log_probs,
                    dim=0,
                ).double()

                proposal_log_probs = (
                    valid_model_log_probs - incremental_log_weight
                )

                sampled_position = torch.multinomial(
                    proposal_log_probs.exp(),
                    num_samples=1,
                ).item()
                token_id = int(valid_tokens[sampled_position].item())

                sampled_tokens[particle] = token_id

                log_weights[particle] += incremental_log_weight
                trajectory_log_weights[particle] += incremental_log_weight

                increment = int(token_id == eoy_idx)
                year_counters[particle] += increment
                done[particle] = year_counters[particle] > year_limits[particle]

            next_tokens = torch.full(
                (nb_particles, 1),
                tokenizer.pad_token_id,
                dtype=input_ids.dtype,
                device=device,
            )
            next_attention = torch.zeros(
                (nb_particles, 1),
                dtype=attention_mask.dtype,
                device=device,
            )

            for particle, token_id in sampled_tokens.items():
                next_tokens[particle, 0] = token_id
                next_attention[particle, 0] = 1

            input_ids = torch.cat([input_ids, next_tokens], dim=1)
            attention_mask = torch.cat([attention_mask, next_attention], dim=1)

            normalized_weights = torch.softmax(log_weights, dim=0)
            ess = 1.0 / normalized_weights.square().sum()

            if verbose:
                print(
                    f"step={step}, "
                    f"done={done.sum().item()}/{nb_particles}, "
                    f"ESS={ess.item():.3f}/{nb_particles}"
                )

            if ess < ess_threshold * nb_particles and not done.all():
                ancestors = torch.multinomial(
                    normalized_weights,
                    num_samples=nb_particles,
                    replacement=True,
                )

                input_ids = input_ids[ancestors].clone()
                attention_mask = attention_mask[ancestors].clone()
                done = done[ancestors].clone()
                year_counters = year_counters[ancestors].clone()
                year_limits = year_limits[ancestors].clone()
                trajectory_log_weights = trajectory_log_weights[ancestors].clone()

                log_weights.zero_()

                if verbose:
                    print(f"step={step}: resampled")

        final_weights = torch.softmax(log_weights, dim=0)

        return (
            input_ids,
            attention_mask,
            final_weights,
            trajectory_log_weights,
        )