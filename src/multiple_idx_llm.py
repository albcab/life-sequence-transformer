import logging
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from hydra.utils import get_class, instantiate
from pytorch_lightning import seed_everything
from tqdm.auto import tqdm

from src.llm.lexer import LEXER
from src.llm.life_sequence_pda import create_pda, get_valid_token_ids


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


def hf_to_l2v(generated, tokenizer, token2index, max_length):
    texts = tokenizer.batch_decode(
        generated,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )

    rows = np.zeros(
        (len(texts), max_length),
        dtype=int,
    )

    for row, text in enumerate(texts):
        tokens = text.strip().split()
        indices = [token2index[token] for token in tokens]
        rows[row, :len(indices)] = indices

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
            strict=False,
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
    # PDA / lexer.
    # ------------------------------------------------------------

    max_stack_height = cfg.generate.sampler.max_stack_height
    max_steps = cfg.generate.sampler.max_steps

    pda = create_pda(tokenizer.eos_token)

    valid_token_ids = get_valid_token_ids(
        pda,
        tokenizer,
    )

    print(
        f"Valid tokenizer tokens: "
        f"{len(valid_token_ids)}/{len(tokenizer)}"
    )

    model_name_short = (
        cfg.model.hparams.model_name
        .split("/")[-1]
        .replace("-", "_")
        .replace(".", "_")
    )

    precomputation_path = (
        f"src/llm/"
        f"{model_name_short}_life_sequence_"
        f"precomputation_max_height_{max_stack_height}.json"
    )

    log.info(
        "Loading PDA precomputation from %s",
        precomputation_path,
    )

    pda.precompute_configurations(
        tokenizer=tokenizer,
        max_height=max_stack_height,
        load_path=precomputation_path,
        eos=tokenizer.eos_token,
    )

    lexer = LEXER(
        pda,
        tokenizer,
    )

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

    eoy_idx = tokenizer.convert_tokens_to_ids(
        "[EOY]"
    )

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

    samples_per_id = int(cfg.datamodule.batch_size)

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
                original_sequence = batch["original_sequence"][0]
                known = batch["padding_mask"][0]

            # --------------------------------------------
            # Extract the single LLM prompt.
            # --------------------------------------------

            batch = model.transfer_batch_to_device(
                batch,
                model.device,
                dataloader_idx=0,
            )

            prompt_input_ids = batch[
                "input_ids"
            ][0:1]

            prompt_attention_mask = batch[
                "attention_mask"
            ][0:1]

            prompt_length = int(
                prompt_attention_mask.sum().item()
            )

            prompt_input_ids = (
                prompt_input_ids[
                    :,
                    :prompt_length,
                ]
            )

            prompt_attention_mask = (
                prompt_attention_mask[
                    :,
                    :prompt_length,
                ]
            )

            # --------------------------------------------
            # SMC.
            # --------------------------------------------

            (
                generated_hf,
                _,
                final_weights,
                trajectory_log_weights,
            ) = model.smc_sample(
                input_ids=prompt_input_ids,
                attention_mask=prompt_attention_mask,
                automata=pda,
                tokenizer=tokenizer,
                lexer=lexer,
                num_years=(
                    cfg.generate.sampler.num_years
                    or trunc_year
                ),
                eoy_idx=eoy_idx,
                valid_token_ids=valid_token_ids,
                nb_particles=samples_per_id,
                max_new_tokens=(
                    cfg.generate.sampler.max_new_tokens
                ),
                ess_threshold=(
                    cfg.generate.sampler.ess_threshold
                ),
                max_stack_height=max_stack_height,
                max_steps=max_steps,
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
            )

            rows[:, known] = 0

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
                original_sequence,
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