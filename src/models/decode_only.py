import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pathlib import Path
import logging
from typing import Optional

"""Custom code"""
from src.models.transformer_utils import ReZero
from src.models.transformer import Performer, NextTokenDecoder
from src.models.dfa import LIFESEQUENCEDFA, evaluate_matcher, evaluate_matcher_batch

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
            batch=batch,
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
            batch=batch,
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
            "pos_emb",
            "layer_pos_emb",
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

        self.accuracy = torchmetrics.Accuracy(
            task="multiclass",
            threshold=0.2,
            num_classes=self.num_outputs,
            average="macro",
            multidim_average="global",
            ignore_index=0,
            top_k=top_k,
        )

        self.accuracy1 = torchmetrics.Accuracy(
            task="multiclass",
            threshold=0.5,
            num_classes=self.num_outputs,
            average="macro",
            multidim_average="global",
            ignore_index=0,
            top_k=1,
        )

        self.precision = torchmetrics.Precision(
            task="multiclass",
            threshold=0.2,
            num_classes=self.num_outputs,
            average="macro",
            multidim_average="global",
            ignore_index=0,
            top_k=top_k,
        )

        self.precision1 = torchmetrics.Precision(
            task="multiclass",
            threshold=0.5,
            num_classes=self.num_outputs,
            average="macro",
            multidim_average="global",
            ignore_index=0,
            top_k=1,
        )

        self.recall = torchmetrics.Recall(
            task="multiclass",
            threshold=0.2,
            num_classes=self.num_outputs,
            average="macro",
            multidim_average="global",
            ignore_index=0,
            top_k=top_k,
        )

        self.recall1 = torchmetrics.Recall(
            task="multiclass",
            threshold=0.2,
            num_classes=self.num_outputs,
            average="macro",
            multidim_average="global",
            ignore_index=0,
            top_k=1,
        )

        self.f1 = torchmetrics.F1Score(
            task="multiclass",
            threshold=0.2,
            num_classes=self.num_outputs,
            average="macro",
            multidim_average="global",
            ignore_index=0,
            top_k=top_k,
        )

        self.f11 = torchmetrics.F1Score(
            task="multiclass",
            threshold=0.5,
            num_classes=self.num_outputs,
            average="macro",
            multidim_average="global",
            ignore_index=0,
            top_k=1,
        )

    def log_metrics(
        self,
        predictions,
        targets,
        loss,
        stage,
        batch=None,
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
                self.accuracy(predictions, targets),
                on_step=on_step,
                on_epoch=on_epoch,
            )
            self.log(
                "train/accuracy1",
                self.accuracy1(predictions, targets),
                on_step=on_step,
                on_epoch=on_epoch,
            )
            self.log(
                "train/recall",
                self.recall(predictions, targets),
                on_step=on_step,
                on_epoch=on_epoch,
            )
            self.log(
                "train/recall1",
                self.recall1(predictions, targets),
                on_step=on_step,
                on_epoch=on_epoch,
            )
            self.log(
                "train/precision",
                self.precision(predictions, targets),
                on_step=on_step,
                on_epoch=on_epoch,
            )
            self.log(
                "train/precision1",
                self.precision1(predictions, targets),
                on_step=on_step,
                on_epoch=on_epoch,
            )
            self.log(
                "train/f1",
                self.f1(predictions, targets),
                on_step=on_step,
                on_epoch=on_epoch,
            )
            self.log(
                "train/f11",
                self.f11(predictions, targets),
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
                self.accuracy(predictions, targets),
                on_step=on_step,
                on_epoch=on_epoch,
            )
            self.log(
                "val/accuracy1",
                self.accuracy1(predictions, targets),
                on_step=on_step,
                on_epoch=on_epoch,
            )
            self.log(
                "val/recall",
                self.recall(predictions, targets),
                on_step=on_step,
                on_epoch=on_epoch,
            )
            self.log(
                "val/recall1",
                self.recall1(predictions, targets),
                on_step=on_step,
                on_epoch=on_epoch,
            )
            self.log(
                "val/precision",
                self.precision(predictions, targets),
                on_step=on_step,
                on_epoch=on_epoch,
            )
            self.log(
                "val/precision1",
                self.precision1(predictions, targets),
                on_step=on_step,
                on_epoch=on_epoch,
            )
            self.log(
                "val/f1",
                self.f1(predictions, targets),
                on_step=on_step,
                on_epoch=on_epoch,
            )
            self.log(
                "val/f11",
                self.f11(predictions, targets),
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
                self.accuracy(predictions, targets),
                on_step=on_step,
                on_epoch=on_epoch,
            )
            self.log(
                "test/accuracy1",
                self.accuracy1(predictions, targets),
                on_step=on_step,
                on_epoch=on_epoch,
            )
            self.log(
                "test/recall",
                self.recall(predictions, targets),
                on_step=on_step,
                on_epoch=on_epoch,
            )
            self.log(
                "test/recall1",
                self.recall1(predictions, targets),
                on_step=on_step,
                on_epoch=on_epoch,
            )
            self.log(
                "test/precision",
                self.precision(predictions, targets),
                on_step=on_step,
                on_epoch=on_epoch,
            )
            self.log(
                "test/precision1",
                self.precision1(predictions, targets),
                on_step=on_step,
                on_epoch=on_epoch,
            )
            self.log(
                "test/f1",
                self.f1(predictions, targets),
                on_step=on_step,
                on_epoch=on_epoch,
            )
            self.log(
                "test/f11",
                self.f11(predictions, targets),
                on_step=on_step,
                on_epoch=on_epoch,
            )


class GeneratorDecoderOnly(TransformerDecoderOnly):

    def __init__(self, hparams, known_years_eval=[1, 5, 10]):
        super(GeneratorDecoderOnly, self).__init__(hparams)
        self.known_years_eval = known_years_eval

    @torch.inference_mode()
    def sample(
        self,
        batch,
        num_years,
        temp=0.8,
        verbose=False,
        eoy_idx=3,
        ending_idx=None,
    ):
        # self.eval()  # Ensure dropout and other train-time behaviors are off

        device = batch["input_ids"].device
        input_ids = batch["input_ids"]  # (B, 3, T)
        padding_mask = batch["padding_mask"]  # (B, T)

        if verbose:
            print(f"Start: predicting users with ids {batch["sequence_id"].tolist()}")
            print(f"Start: conditioning on {padding_mask.sum(dim=1).tolist()} tokens.")

        B, _, max_len = input_ids.shape
        bidx = torch.arange(B)
        generated_mask = torch.zeros_like(padding_mask)

        year_counters = torch.zeros(B, dtype=torch.long, device=device)
        done = torch.zeros(B, dtype=torch.bool, device=device)
        if ending_idx is not None:
            end_counter = torch.zeros(B, dtype=torch.long, device=device)

        for i in range(max_len):

            _next_pos = padding_mask.sum(dim=1)
            next_pos = _next_pos.clamp(max=max_len - 1) 

            prev_tokens = input_ids[bidx, 0, next_pos - 1]
            prev_years = input_ids[bidx, 1, next_pos - 1]
            prev_ages = input_ids[bidx, 2, next_pos - 1]

            increment = (prev_tokens == eoy_idx).int()
            curr_years = prev_years + increment
            curr_ages = prev_ages + increment

            year_counters += increment
            if ending_idx is not None:
                end_counter += (prev_tokens == ending_idx).int()
                done = (end_counter > num_years) | (_next_pos >= max_len)
            else:
                done = (year_counters > num_years) | (_next_pos >= max_len)

            if verbose:
                print(f"Step _{i}: there are {increment.sum().item()} [EOY] tokens")
                print(f"Step _{i}: {done.int().sum().item()} sequences are done")

            if done.all():
                break

            logits = self({
                "input_ids": input_ids,
                "padding_mask": padding_mask,
            })  # (B, T, vocab_size)

            last_logits = logits[bidx, next_pos - 1]  # (B, vocab_size)
            probs = F.softmax(last_logits / temp, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1).double()

            in_bounds = _next_pos < max_len
            idx_in_bounds = torch.arange(B, device=device)[in_bounds]
            next_pos_in_bounds = _next_pos[in_bounds]

            input_ids[idx_in_bounds, 0, next_pos_in_bounds] = next_tokens[in_bounds]
            input_ids[idx_in_bounds, 1, next_pos_in_bounds] = curr_years[in_bounds]
            input_ids[idx_in_bounds, 2, next_pos_in_bounds] = curr_ages[in_bounds]
            padding_mask[idx_in_bounds, next_pos_in_bounds] = 1
            generated_mask[idx_in_bounds, next_pos_in_bounds] = 1

            if verbose:
                # print(f"Step {i}_: {generated_mask.sum(dim=1).tolist()} tokens generated.")
                print(f"Step {i}_: {curr_years.tolist()} current years.")
                print(f"Step {i}_: {curr_ages.tolist()} current ages.")

        # self.train() 

        batch["input_ids"] = input_ids
        batch["padding_mask"] = padding_mask
        return batch, generated_mask

    def validate_tokens_batched(self, indices, configurations, dfa, remaining_tokens):
        # indices = np.asarray(indices, dtype=np.int64)
        validated_indices = {}

        for state, prefix, distance in configurations:
            # Terminal states need not have outgoing transitions.
            for matcher, next_state in dfa.transitions_by_state.get(state, ()):
                satisfied, extendable, fixable = evaluate_matcher_batch(matcher, prefix, indices)

                completed = satisfied & ~extendable
                continuing = satisfied & extendable
                partial = ~satisfied & fixable

                if dfa.state_distances[next_state] <= remaining_tokens:
                    for index in indices[completed | continuing]:
                        index = int(index)
                        validated_indices.setdefault(index, set()).add((next_state, (), dfa.state_distances[next_state]))

                if distance <= remaining_tokens:
                    for index in indices[continuing]:
                        index = int(index)
                        validated_indices.setdefault(index, set()).add((state, prefix + (index,), distance))

                for index in indices[partial]:
                    index = int(index)
                    new_distance = distance - len(prefix) - 1

                    if new_distance <= remaining_tokens:
                        validated_indices.setdefault(index, set()).add((state, prefix + (index,), new_distance))

        return validated_indices

    def validate_tokens(self, top_indices, configurations, dfa, remaining_tokens, beam_width):
        count_valid = 0
        count_topk = 0
        validated_top_indices = {} # {index: {(state, prefix, distance), ...}, ...}
        for index in top_indices:
            index = index.item() if hasattr(index, "item") else int(index)
            count_topk += 1
            if count_topk > beam_width and count_valid > 0:
                break
            for state, prefix, distance in configurations:
                for (current_state, matcher), next_state in dfa.transitions.items():
                    if state == current_state:
                        satisfied, extendable, fixable = evaluate_matcher(matcher=matcher, tokens=prefix + (index,))
                        # case 1: satisfied = True, extendable = False, fixable = False. Add the new configuration (next_state, (), dfa.state_distances[next_state]) if dfa.state_distances[next_state] <= max_len - step + 1
                        if satisfied and not extendable and not fixable:
                            if dfa.state_distances[next_state] <= remaining_tokens:
                                if index not in validated_top_indices:
                                    validated_top_indices[index] = set()
                                validated_top_indices[index].add((next_state, (), dfa.state_distances[next_state]))
                                count_valid += 1
                        # case 2: satisfied = True, extendable = True, fixable = False. Add two new configurations: (next_state, (), dfa.state_distances[next_state]) and (current_state, prefix + (index,), distance) if for both dfa.state_distances[next_state] <= max_len - step + 1 and distance < max_len - step + 1 respectively
                        elif satisfied and extendable and not fixable:
                            if dfa.state_distances[next_state] <= remaining_tokens:
                                if index not in validated_top_indices:
                                    validated_top_indices[index] = set()
                                validated_top_indices[index].add((next_state, (), dfa.state_distances[next_state]))
                                count_valid += 1
                            if distance <= remaining_tokens:
                                if index not in validated_top_indices:
                                    validated_top_indices[index] = set()
                                validated_top_indices[index].add((current_state, prefix + (index,), distance))
                                count_valid += 1
                        # case 3: satisfied = False, extendable = False, fixable = True. Add the new configuration (current_state, prefix + (index,), distance - len(prefix + (index,))) if( distance - len(prefix + (index,))) <= max_len - step + 1
                        elif not satisfied and not extendable and fixable:
                            if (distance - len(prefix + (index,))) <= remaining_tokens:
                                if index not in validated_top_indices:
                                    validated_top_indices[index] = set()
                                validated_top_indices[index].add((current_state, prefix + (index,), distance - len(prefix + (index,))))
                                count_valid += 1
                        # case 4: satisfied = False, extendable = False, fixable = False. Do not add any new configuration
                        elif not satisfied and not extendable and not fixable:
                            pass
        return validated_top_indices
    
    @torch.inference_mode()
    def beam_search(
        self,
        batch,
        num_years: int,
        beam_width: int = 5,
        length_penalty: float = 1.0,
        max_len: Optional[int] = None,
        eoy_idx: int = 3,
        ending_idx: Optional[int] = None,
        verbose: bool = False,
    ):
        device = batch["input_ids"].device
        B, _, max_seq_len = batch["input_ids"].shape
        if max_len is None:
            max_len = max_seq_len

        if isinstance(num_years, (int, float)):
            num_years_by_sample = [int(num_years)] * B
        elif isinstance(num_years, torch.Tensor):
            num_years_by_sample = [int(value) for value in num_years.tolist()]
        else:
            num_years_by_sample = [int(value) for value in num_years]
        if len(num_years_by_sample) != B:
            raise ValueError("num_years must be a scalar or contain one value per sample")

        dfa = LIFESEQUENCEDFA()
        if max_len < dfa.state_distances[dfa.initial_state]:
            raise ValueError(f"max_len must be at least {dfa.state_distances[dfa.initial_state]} to allow any valid sequence generation.")
        
        best_input_ids = batch["input_ids"].clone()
        best_padding_mask = batch["padding_mask"].clone()
        best_generated_mask = torch.zeros_like(best_padding_mask)

        beams_by_sample = []
        for b in range(B):
            input_ids = batch["input_ids"][b:b+1].clone()  # (1, 3, T)
            padding_mask = batch["padding_mask"][b:b+1].clone()  # (1, T)
            initial_sequence = []
            for m in range(len(padding_mask[0])):
                if not padding_mask[0][m].item(): break
                initial_sequence.append(int(input_ids[0][0][m].item()))

            generated_mask = torch.zeros_like(padding_mask)

            # print("Init seq", initial_sequence)
            configs = dfa.get_initial_conf(initial_sequence, verbose=verbose)

            beams = []
            for s, p in configs:
                # (input_ids, padding_mask, generated_mask, year_counter, log_prob, done, {(state, prefix, distance)})
                beams.append((input_ids, padding_mask, generated_mask, 0, 0.0, False, {(s, p, dfa.state_distances[s])}))
            beams_by_sample.append(beams)

        # At each generation step, run every active beam from every sample through
        # the transformer in one forward pass. DFA configurations and pruning remain
        # independent, so this changes throughput rather than search semantics.
        for step in range(max_len):
            remaining_tokens = max_len - step + 1
            active_beams = []
            candidates_by_sample = [[] for _ in range(B)]

            for b, beams in enumerate(beams_by_sample):
                for beam in beams:
                    inp_ids, pad_mask, gen_mask, year_cnt, log_prob, done, configurations = beam
                    if done:
                        candidates_by_sample[b].append(beam)
                        continue

                    next_pos = int(pad_mask.sum().item())
                    if next_pos >= max_len:
                        candidates_by_sample[b].append(
                            (inp_ids, pad_mask, gen_mask, year_cnt, log_prob, True, configurations)
                        )
                        continue

                    active_beams.append((b, beam, next_pos))

            if active_beams:
                active_input_ids = torch.cat(
                    [beam[1][0] for beam in active_beams], dim=0
                )
                active_padding_masks = torch.cat(
                    [beam[1][1] for beam in active_beams], dim=0
                )
                logits = self({
                    "input_ids": active_input_ids,
                    "padding_mask": active_padding_masks,
                })

                row_indices = torch.arange(len(active_beams), device=device)
                next_positions = torch.tensor(
                    [item[2] - 1 for item in active_beams], device=device
                )
                last_logits = logits[row_indices, next_positions]
                probs = F.softmax(last_logits, dim=-1)
                sorted_indices = torch.argsort(probs, dim=-1, descending=True)

                # One device synchronization per decoding step avoids a synchronization
                # for every candidate token inspected by the DFA.
                probs_cpu = probs.detach().cpu()
                sorted_indices_cpu = sorted_indices.detach().cpu()

                for row, (b, beam, next_pos) in enumerate(active_beams):
                    inp_ids, pad_mask, gen_mask, year_cnt, log_prob, _, configurations = beam
                    validated_top_indices = self.validate_tokens(
                        sorted_indices_cpu[row],
                        configurations,
                        dfa,
                        remaining_tokens,
                        beam_width,
                    )

                    for key, item in validated_top_indices.items():
                        new_token = key
                        new_log_prob = log_prob + torch.log(probs_cpu[row, key]).item()

                        increment = 1 if new_token == eoy_idx else 0
                        new_year_cnt = year_cnt + increment

                        done_flag = False
                        if ending_idx is not None and new_token == ending_idx:
                            done_flag = True
                        elif new_year_cnt > num_years_by_sample[b]:
                            done_flag = True
                        elif next_pos + 1 >= max_len:
                            done_flag = True

                        new_inp_ids = inp_ids.clone()
                        new_pad_mask = pad_mask.clone()
                        new_gen_mask = gen_mask.clone()

                        new_inp_ids[0, 0, next_pos] = new_token
                        prev_year = inp_ids[0, 1, next_pos - 1].item()
                        prev_age = inp_ids[0, 2, next_pos - 1].item()
                        new_year = prev_year + increment
                        new_age = prev_age + increment
                        new_inp_ids[0, 1, next_pos] = new_year
                        new_inp_ids[0, 2, next_pos] = new_age

                        new_pad_mask[0, next_pos] = 1
                        new_gen_mask[0, next_pos] = 1

                        candidates_by_sample[b].append(
                            (new_inp_ids, new_pad_mask, new_gen_mask, new_year_cnt, new_log_prob, done_flag, item)
                        )

            if not active_beams:
                break

            for b, candidates in enumerate(candidates_by_sample):
                if not candidates:
                    raise RuntimeError(
                        f"DFA beam search found no valid continuation for batch item {b}"
                    )
                candidates.sort(
                    key=lambda x: x[4] / (x[1].sum().item() ** length_penalty),
                    reverse=True,
                )
                beams_by_sample[b] = candidates[:beam_width]

        for b, beams in enumerate(beams_by_sample):
            best_beam = max(beams, key=lambda x: x[4] / (x[1].sum().item() ** length_penalty))
            best_input_ids[b:b+1] = best_beam[0]
            best_padding_mask[b:b+1] = best_beam[1]
            best_generated_mask[b:b+1] = best_beam[2]

        batch["input_ids"] = best_input_ids
        batch["padding_mask"] = best_padding_mask
        return batch, best_generated_mask

    @torch.inference_mode()
    def smc_sample(
        self,
        batch,
        num_years,
        ess_threshold=0.5,
        verbose=False,
        eoy_idx=3,
        ending_idx=2,
    ):
        """Sample weighted lives, retaining ended particles as absorbing states.

        The first EOL is scored normally. Finished particles receive no further
        tokens or importance increments, but remain eligible for resampling.
        Preserve the existing EOY-count stopping convention (> num_years).
        """
        device = batch["input_ids"].device
        input_ids = batch["input_ids"].clone()
        padding_mask = batch["padding_mask"].clone()

        B, _, max_len = input_ids.shape
        vocab_size = self.num_outputs

        if not 0 < ess_threshold <= 1:
            raise ValueError("ess_threshold must be in (0, 1]")

        dfa = LIFESEQUENCEDFA()
        indices = np.arange(vocab_size, dtype=np.int64)

        generated_mask = torch.zeros_like(padding_mask)
        year_counters = torch.zeros(B, dtype=torch.long, device=device)
        done = torch.zeros(B, dtype=torch.bool, device=device)

        # Weights accumulated since the most recent resampling.
        log_weights = torch.zeros(B, dtype=torch.float64, device=device)

        # Cumulative importance correction associated with each trajectory.
        trajectory_log_weights = torch.zeros(B, dtype=torch.float64, device=device)

        configurations = []

        for b in range(B):
            length = int(padding_mask[b].sum().item())
            initial_sequence = input_ids[b, 0, :length].long().cpu().tolist()
            configs = dfa.get_initial_conf(initial_sequence, verbose=False)

            configurations.append({(state, prefix, dfa.state_distances[state]) for state, prefix in configs})

            if ending_idx is not None and initial_sequence and initial_sequence[-1] == ending_idx:
                done[b] = True

        for step in range(max_len):
            next_positions = padding_mask.sum(dim=1).long()
            done |= next_positions >= max_len

            if done.all():
                break

            active = torch.where(~done)[0]
            active_next_positions = next_positions[active]

            logits = self({
                "input_ids": input_ids[active],
                "padding_mask": padding_mask[active],
            })

            rows = torch.arange(len(active), device=device)
            next_logits = logits[rows, active_next_positions - 1]

            # Log probabilities under the original model, normalized
            # over the entire vocabulary.
            model_log_probs = F.log_softmax(next_logits, dim=-1)

            for row, b_tensor in enumerate(active):
                b = int(b_tensor.item())
                next_pos = int(active_next_positions[row].item())
                remaining_tokens = max_len - next_pos

                validated = self.validate_tokens_batched(
                    indices,
                    configurations[b],
                    dfa,
                    remaining_tokens,
                )

                if not validated:
                    raise RuntimeError(f"DFA found no valid continuation for particle {b}")

                valid_tokens = torch.tensor(list(validated.keys()), dtype=torch.long, device=device)

                # Original model probabilities for valid tokens.
                valid_model_log_probs = model_log_probs[row, valid_tokens]

                incremental_log_weight = torch.logsumexp(valid_model_log_probs, dim=0).double()

                proposal_log_probs = valid_model_log_probs - incremental_log_weight
                proposal_probs = proposal_log_probs.exp()

                sampled_position = torch.multinomial(
                    proposal_probs,
                    num_samples=1,
                ).item()

                new_token = int(valid_tokens[sampled_position].item())

                log_weights[b] += incremental_log_weight
                trajectory_log_weights[b] += incremental_log_weight

                configurations[b] = validated[new_token]

                increment = int(new_token == eoy_idx)
                year_counters[b] += increment

                prev_year = input_ids[b, 1, next_pos - 1]
                prev_age = input_ids[b, 2, next_pos - 1]

                input_ids[b, 0, next_pos] = new_token
                input_ids[b, 1, next_pos] = prev_year + increment
                input_ids[b, 2, next_pos] = prev_age + increment

                padding_mask[b, next_pos] = 1
                generated_mask[b, next_pos] = 1

                done[b] = year_counters[b] > num_years
                if ending_idx is not None and new_token == ending_idx:
                    done[b] = True

                if next_pos + 1 >= max_len:
                    done[b] = True

            # Normalized SMC weights.
            normalized_weights = F.softmax(log_weights, dim=0)

            # ESS = 1 / sum(w_i^2)
            ess = 1.0 / normalized_weights.square().sum()

            if verbose:
                print(
                    f"Step {step}: "
                    f"{done.sum().item()}/{B} particles done, "
                    f"ESS={ess.item():.2f}/{B}"
                )

            if ess < ess_threshold * B and not done.all():
                ancestors = torch.multinomial(
                    normalized_weights,
                    num_samples=B,
                    replacement=True,
                )

                input_ids = input_ids[ancestors].clone()
                padding_mask = padding_mask[ancestors].clone()
                generated_mask = generated_mask[ancestors].clone()
                year_counters = year_counters[ancestors].clone()
                done = done[ancestors].clone()
                trajectory_log_weights = trajectory_log_weights[ancestors].clone()

                configurations = [
                    configurations[int(a.item())].copy()
                    for a in ancestors
                ]

                # Equal weights immediately after resampling.
                log_weights.zero_()

                print(f"Step {step}: resampled particles")

        batch["input_ids"] = input_ids
        batch["padding_mask"] = padding_mask

        # Final normalized SMC weights since the last resampling.
        final_weights = F.softmax(log_weights, dim=0)

        return batch, generated_mask, final_weights, trajectory_log_weights

    def configure_optimizers(self):
        """Configuration of the Optimizer and the Learning Rate Scheduler."""
        no_decay = [
            "bias",
            "norm",
            "age",
            "year",
            "token",
            "pos_emb",
            "layer_pos_emb",
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

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ExponentialLR(
                    optimizer, gamma=self.hparams.lr_gamma
                ), 
                "interval": "epoch",
                "frequency": 1,
                "name": "learning_rate",
            },
        }


    def init_metrics(self):

        self.accuracy = torchmetrics.Accuracy(
            task="multiclass",
            threshold=0.2,
            num_classes=self.num_outputs,
            average="macro",
            multidim_average="global",
            ignore_index=0,
            top_k=5,
        )

        self.accuracy1 = torchmetrics.Accuracy(
            task="multiclass",
            threshold=0.5,
            num_classes=self.num_outputs,
            average="macro",
            multidim_average="global",
            ignore_index=0,
            top_k=1,
        )

        self.precision = torchmetrics.Precision(
            task="multiclass",
            threshold=0.2,
            num_classes=self.num_outputs,
            average="macro",
            multidim_average="global",
            ignore_index=0,
            top_k=5,
        )

        self.precision1 = torchmetrics.Precision(
            task="multiclass",
            threshold=0.5,
            num_classes=self.num_outputs,
            average="macro",
            multidim_average="global",
            ignore_index=0,
            top_k=1,
        )

        self.recall = torchmetrics.Recall(
            task="multiclass",
            threshold=0.2,
            num_classes=self.num_outputs,
            average="macro",
            multidim_average="global",
            ignore_index=0,
            top_k=5,
        )

        self.recall1 = torchmetrics.Recall(
            task="multiclass",
            threshold=0.2,
            num_classes=self.num_outputs,
            average="macro",
            multidim_average="global",
            ignore_index=0,
            top_k=1,
        )

        self.f1 = torchmetrics.F1Score(
            task="multiclass",
            threshold=0.2,
            num_classes=self.num_outputs,
            average="macro",
            multidim_average="global",
            ignore_index=0,
            top_k=5,
        )

        self.f11 = torchmetrics.F1Score(
            task="multiclass",
            threshold=0.5,
            num_classes=self.num_outputs,
            average="macro",
            multidim_average="global",
            ignore_index=0,
            top_k=1,
        )

    def log_metrics(
        self,
        predictions,
        targets,
        loss,
        stage,
        batch=None,
        on_step: bool = True,
        on_epoch: bool = True,
    ):
        """Compute on step/epoch metrics"""
        loss = loss.cpu()
        predicts = predictions
        predictions = F.softmax(predictions, dim=-1).permute(0, 2, 1)

        if stage == "train":

            self.log("train/loss", loss, on_step=on_step, on_epoch=on_epoch)
            self.log(
                "train/perplexity", torch.sqrt(loss), on_step=on_step, on_epoch=on_epoch
            )
            self.log(
                "train/accuracy",
                self.accuracy(predictions, targets),
                on_step=on_step,
                on_epoch=on_epoch,
            )
            self.log(
                "train/accuracy1",
                self.accuracy1(predictions, targets),
                on_step=on_step,
                on_epoch=on_epoch,
            )
            self.log(
                "train/recall",
                self.recall(predictions, targets),
                on_step=on_step,
                on_epoch=on_epoch,
            )
            self.log(
                "train/recall1",
                self.recall1(predictions, targets),
                on_step=on_step,
                on_epoch=on_epoch,
            )
            self.log(
                "train/precision",
                self.precision(predictions, targets),
                on_step=on_step,
                on_epoch=on_epoch,
            )
            self.log(
                "train/precision1",
                self.precision1(predictions, targets),
                on_step=on_step,
                on_epoch=on_epoch,
            )
            self.log(
                "train/f1",
                self.f1(predictions, targets),
                on_step=on_step,
                on_epoch=on_epoch,
            )
            self.log(
                "train/f11",
                self.f11(predictions, targets),
                on_step=on_step,
                on_epoch=on_epoch,
            )


        elif stage == "val" or stage == "test":

            self.log(f"{stage}/loss", loss, on_step=on_step, on_epoch=on_epoch)
            self.log(
                f"{stage}/perplexity", torch.sqrt(loss), on_step=on_step, on_epoch=on_epoch
            )
            self.log(
                f"{stage}/accuracy",
                self.accuracy(predictions, targets),
                on_step=on_step,
                on_epoch=on_epoch,
            )
            self.log(
                f"{stage}/accuracy1",
                self.accuracy1(predictions, targets),
                on_step=on_step,
                on_epoch=on_epoch,
            )
            self.log(
                f"{stage}/recall",
                self.recall(predictions, targets),
                on_step=on_step,
                on_epoch=on_epoch,
            )
            self.log(
                f"{stage}/recall1",
                self.recall1(predictions, targets),
                on_step=on_step,
                on_epoch=on_epoch,
            )
            self.log(
                f"{stage}/precision",
                self.precision(predictions, targets),
                on_step=on_step,
                on_epoch=on_epoch,
            )
            self.log(
                f"{stage}/precision1",
                self.precision1(predictions, targets),
                on_step=on_step,
                on_epoch=on_epoch,
            )
            self.log(
                f"{stage}/f1",
                self.f1(predictions, targets),
                on_step=on_step,
                on_epoch=on_epoch,
            )
            self.log(
                f"{stage}/f11",
                self.f11(predictions, targets),
                on_step=on_step,
                on_epoch=on_epoch,
            )

            yearseq = batch["input_ids"].long()[:, 1, :].detach()
            padding_mask = batch["padding_mask"].detach().bool()
            min_years = yearseq[:, 6].reshape(-1, 1)

            for a in self.known_years_eval:
                start_years = min_years + a
                start_indx = ((yearseq >= start_years) | (~padding_mask)).int().argmax(axis=1)
                _targets = torch.zeros_like(targets)
                for b, indx in enumerate(start_indx):
                    _targets[b, indx:] = targets[b, indx:]

                _loss = self.loss_fn(predicts.permute(0, 2, 1), target=_targets)
                self.log(
                    f"{stage}/perplexity_post{a}", torch.sqrt(_loss), on_step=on_step, on_epoch=on_epoch
                )

                self.log(
                    f"{stage}/accuracy_post{a}",
                    self.accuracy(predictions, _targets),
                    on_step=on_step,
                    on_epoch=on_epoch,
                )
                self.log(
                    f"{stage}/accuracy1_post{a}",
                    self.accuracy1(predictions, _targets),
                    on_step=on_step,
                    on_epoch=on_epoch,
                )
                self.log(
                    f"{stage}/recall_post{a}",
                    self.recall(predictions, _targets),
                    on_step=on_step,
                    on_epoch=on_epoch,
                )
                self.log(
                    f"{stage}/recall1_post{a}",
                    self.recall1(predictions, _targets),
                    on_step=on_step,
                    on_epoch=on_epoch,
                )
                self.log(
                    f"{stage}/precision_post{a}",
                    self.precision(predictions, _targets),
                    on_step=on_step,
                    on_epoch=on_epoch,
                )
                self.log(
                    f"{stage}/precision1_post{a}",
                    self.precision1(predictions, _targets),
                    on_step=on_step,
                    on_epoch=on_epoch,
                )
                self.log(
                    f"{stage}/f1_post{a}",
                    self.f1(predictions, _targets),
                    on_step=on_step,
                    on_epoch=on_epoch,
                )
                self.log(
                    f"{stage}/f11_post{a}",
                    self.f11(predictions, _targets),
                    on_step=on_step,
                    on_epoch=on_epoch,
                )
