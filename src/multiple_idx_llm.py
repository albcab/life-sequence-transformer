import logging
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from hydra.utils import get_class, instantiate
from pytorch_lightning import seed_everything
from tqdm.auto import tqdm



HOME_PATH = str(Path())
log = logging.getLogger(__name__)


def last_ckpt(dir_):
    print(HOME_PATH)
    ckpt_path = Path(HOME_PATH, dir_, "best.ckpt")
    ckpt_ = Path(HOME_PATH, dir_)

    if ckpt_path.exists():
        print("CHECKPOINT EXISTS:", str(ckpt_path))
        return str(ckpt_path)

    if ckpt_.exists():
        print("CHECKPOINT EXISTS:", str(ckpt_))
        return str(ckpt_)

    print("CHECKPOINT DOES NOT EXIST:", str(ckpt_path))
    raise FileNotFoundError(ckpt_path)


def hf_to_l2v(generated, tokenizer, token2index, max_length, attention_mask=None):
    texts = [tokenizer.decode(row if attention_mask is None else row[attention_mask[i].bool()],
                             skip_special_tokens=False, clean_up_tokenization_spaces=False)
             for i, row in enumerate(generated)]
    indices = [[token2index[word] for word in text.split()] for text in texts]
    width = max(max_length, max(map(len, indices)))
    rows = np.zeros((len(indices), width), dtype=int)
    for i, values in enumerate(indices):
        rows[i, :len(values)] = values
    return rows


@hydra.main(
    config_path="../conf",
    config_name="gconfig",
    version_base=None,
)
def main(cfg):
    seed_everything(cfg.seed)

    # ------------------------------------------------------------
    # Tokenizer / data.
    # ------------------------------------------------------------

    tokenizer = instantiate(cfg.tokenizer)
    tokenizer.pad_token = tokenizer.eos_token

    data = instantiate(
        cfg.datamodule,
        tokenizer=tokenizer,
        _convert_="all",
    )

    # ------------------------------------------------------------
    # Model.
    # ------------------------------------------------------------

    ModelClass = get_class(cfg.model._target_)

    if cfg.get("model_path"):
        model = ModelClass.load_from_checkpoint(
            cfg.model_path,
            strict=True,
            hparams=cfg.model.hparams,
            tokenizer=tokenizer,
        )
    else:
        log.info(
            "No Lightning checkpoint provided. "
            "Loading HuggingFace checkpoint: %s",
            cfg.model.hparams.model_name,
        )

        model = ModelClass(
            hparams=cfg.model.hparams,
            tokenizer=tokenizer,
        )

    model.to(cfg.trainer.accelerator)
    model.eval()

    # ------------------------------------------------------------
    # Output.
    # ------------------------------------------------------------

    dir_name = Path(
        "generated",
        cfg.implementation,
        cfg.name,
        cfg.generate.dataloader.file_name.rsplit(".", 1)[0],
        str(cfg.generate.dataloader.offset),
    )

    dir_name.mkdir(
        parents=True,
        exist_ok=True,
    )

    token2index = data.vocabulary.token2index
    index2token = data.vocabulary.index2token

    # ------------------------------------------------------------
    # Users.
    # ------------------------------------------------------------

    ids_df = pd.read_csv(
        cfg.generate.dataloader.file_name
    )

    ids = [
        int(idx)
        for idx in ids_df.USER_ID.tolist()
    ]

    trunc_years = [
        int(
            year
            + cfg.generate.dataloader.offset
        )
        for year in ids_df.year_to_remove.tolist()
    ]

    if len(ids) != len(set(ids)):
        raise ValueError(
            "Generation input must contain unique USER_ID values"
        )

    samples_per_id = int(cfg.generate.sampler.nb_particles)

    if samples_per_id < 1:
        raise ValueError(
            "samples_per_id must be at least 1"
        )

    # ------------------------------------------------------------
    # Generate one user at a time.
    # ------------------------------------------------------------

    for idx, trunc_year in tqdm(
        zip(ids, trunc_years),
        total=len(ids),
        desc="Generating users",
        unit="user",
        dynamic_ncols=True,
    ):
        index_file = (
            dir_name / f"{idx}_index.csv"
        )

        token_file = (
            dir_name / f"{idx}_token.csv"
        )

        weight_file = (
            dir_name / f"{idx}_weights.csv"
        )

        if (
            index_file.exists()
            and token_file.exists()
            and weight_file.exists()
        ):
            tqdm.write(
                f"Files for id={idx} already exist, skipping..."
            )
            continue

        tqdm.write(
            f"Starting id={idx} "
            f"w/o {trunc_year} years..."
        )

        # SMC itself creates the particle population, so we need
        # one truncated prompt for this person.
        assert (
            cfg.generate.dataloader.reps == 1
        ), "SMC works only on one batch per person."

        dataloader = data.single_idx_dataloader(
            idx=idx,
            trunc_years=trunc_year,
            reps=1,
            split=cfg.generate.dataloader.split,
        )

        cum_rows = []
        cum_weights = []
        original_sequence = None
        known = None

        for batch in dataloader:

            if original_sequence is None:
                original_sequence = np.asarray(batch["original_sequence"][0])
                known = np.asarray(batch["padding_mask"][0], dtype=bool)

            # --------------------------------------------
            # Extract the single LLM prompt.
            # --------------------------------------------

            prefix = original_sequence[known]
            prompt = " ".join(index2token[int(token)] for token in prefix)
            encoded = tokenizer.encode(prompt, add_special_tokens=False)
            prompt_input_ids = torch.tensor([encoded], device=model.device)
            prompt_attention_mask = torch.ones_like(prompt_input_ids)

            # --------------------------------------------
            # SMC.
            # --------------------------------------------

            (
                generated_hf,
                generated_attention,
                final_weights,
                trajectory_log_weights,
            ) = model.smc_sample(
                input_ids=prompt_input_ids,
                attention_mask=prompt_attention_mask,
                tokenizer=tokenizer,
                num_years=(
                    cfg.generate.sampler.num_years
                    if cfg.generate.sampler.num_years is not None else trunc_year
                ),
                token2index=token2index,
                nb_particles=samples_per_id,
                max_new_tokens=(
                    cfg.generate.sampler.max_new_tokens
                ),
                ess_threshold=(
                    cfg.generate.sampler.ess_threshold
                ),
                verbose=cfg.generate.sampler.verbose,
            )

            # --------------------------------------------
            # Convert HF tokens back to life-sequence
            # vocabulary indices.
            # --------------------------------------------

            generated_hf = (
                generated_hf
                .detach()
                .cpu()
            )

            rows = hf_to_l2v(
                generated_hf,
                tokenizer,
                token2index,
                cfg.datamodule.task.max_length,
                attention_mask=generated_attention.cpu(),
            )

            rows[:, np.flatnonzero(known)] = 0

            cum_rows.append(rows)

            cum_weights.append(
                final_weights
                .detach()
                .cpu()
                .numpy()
            )

        # --------------------------------------------------------
        # Save immediately after finishing this user.
        # --------------------------------------------------------

        generated_rows = np.vstack(
            cum_rows
        )

        weights = np.concatenate(
            cum_weights
        )

        if len(generated_rows) != samples_per_id:
            raise RuntimeError(
                f"Expected {samples_per_id} generations "
                f"for id={idx}, got {len(generated_rows)}"
            )

        data_rows = np.vstack(
            [
                np.pad(original_sequence, (0, generated_rows.shape[1]-len(original_sequence))),
                generated_rows,
            ]
        )

        np.savetxt(
            index_file,
            data_rows,
            fmt="%d",
            delimiter=",",
        )

        token_rows = np.vectorize(
            lambda i: index2token[i]
        )(data_rows)

        np.savetxt(
            token_file,
            token_rows,
            fmt="%s",
            delimiter=",",
        )

        np.savetxt(
            weight_file,
            weights,
            fmt="%.10g",
        )


if __name__ == "__main__":
    main()