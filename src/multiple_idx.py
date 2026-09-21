import logging
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
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


def save_result(index_file, token_file, original_sequence, generated_rows, index2token):
    data_rows = np.vstack([original_sequence] + generated_rows)
    np.savetxt(index_file, data_rows, fmt="%d", delimiter=",")
    token_rows = np.vectorize(lambda i: index2token[i])(data_rows)
    np.savetxt(token_file, token_rows, fmt="%s", delimiter=",")


@hydra.main(config_path="../conf", config_name="gconfig", version_base=None)
def main(cfg):
    seed_everything(cfg.seed)

    data = instantiate(cfg.datamodule, _convert_="all")
    cfg.model.hparams.vocab_size = data.vocabulary.size()
    print(cfg.model.hparams.vocab_size)
    cfg.model.hparams.n_users = data.corpus.population.data_split().train.shape[0]

    ModelClass = get_class(cfg.model._target_)
    if not cfg.get("model_path"):
        raise NotImplementedError("No pretrained model checkpoint to load")
    model = ModelClass.load_from_checkpoint(
        cfg.model_path,
        strict=False,
        hparams=cfg.model.hparams,
    )

    dir_name = Path(
        "generated",
        cfg.implementation,
        cfg.name,
        cfg.generate.dataloader.file_name.rsplit(".", 1)[0],
        str(cfg.generate.dataloader.offset),
    )
    dir_name.mkdir(parents=True, exist_ok=True)

    token2index = data.vocabulary.token2index
    index2token = data.vocabulary.index2token
    eoy_idx = token2index["[EOY]"]
    eol_idx = token2index["[EOL]"]

    ids_df = pd.read_csv(cfg.generate.dataloader.file_name)
    ids = [int(idx) for idx in ids_df.USER_ID.tolist()]
    trunc_years = [
        int(year + cfg.generate.dataloader.offset)
        for year in ids_df.year_to_remove.tolist()
    ]
    if len(ids) != len(set(ids)):
        raise ValueError("Generation input must contain unique USER_ID values")

    for idx, trunc_year in tqdm(
        zip(ids, trunc_years),
        total=len(ids),
        desc="Generating users",
        unit="user",
        dynamic_ncols=True,
    ):
        index_file = dir_name / f"{idx}_index.csv"
        token_file = dir_name / f"{idx}_token.csv"
        weight_file = dir_name / f"{idx}_weights.csv"

        if index_file.exists() and token_file.exists() and weight_file.exists():
            tqdm.write(f"Files for id={idx} already exist, skipping...")
            continue

        tqdm.write(f"Starting id={idx} w/o {trunc_year} years...")

        assert cfg.generate.dataloader.reps == 1, "SMC works only on one batch per person."
        dataloader = data.single_idx_dataloader(
            idx=idx,
            trunc_years=trunc_year,
            reps=cfg.generate.dataloader.reps * cfg.datamodule.batch_size,
            split=cfg.generate.dataloader.split)

        name = None
        cum_rows = []
        cum_weights = []
        for batch in dataloader:

            if name is None:
                name = batch['sequence_id'][0].item()
                original_sequence = batch['original_sequence'][0].detach().cpu().numpy()
                known = batch['padding_mask'][0].detach().clone().cpu().bool()

            batch = model.transfer_batch_to_device(batch, model.device, dataloader_idx=0)
            
            sample_batch, _, final_weights, trajectory_log_weights = model.smc_sample(
                batch,
                num_years=(
                    trunc_year if cfg.generate.sampler.num_years is None
                    else cfg.generate.sampler.num_years
                ),
                ess_threshold=cfg.generate.sampler.ess_threshold,
                verbose=cfg.generate.sampler.verbose,
                eoy_idx=eoy_idx,
                ending_idx=eol_idx,
            )
            rows = sample_batch['input_ids'][:, 0].detach().cpu().numpy()
            rows[:, known] = 0
            cum_rows.append(rows)
            cum_weights.append(final_weights.detach().cpu().numpy())

        data_rows = np.vstack([original_sequence] + cum_rows)
        weights = np.concatenate(cum_weights)
        np.savetxt(index_file, data_rows, fmt="%d", delimiter=",")
        np.savetxt(token_file, np.vectorize(lambda i: index2token[i])(data_rows), fmt="%s", delimiter=",")
        np.savetxt(weight_file, weights, fmt="%.10g")

""" ## BEAM SEARCH
    pending_ids = []
    pending_trunc_years = []
    output_paths = {}
    for idx, trunc_year in zip(ids, trunc_years):
        index_file = dir_name / f"{idx}_index.csv"
        token_file = dir_name / f"{idx}_token.csv"
        if index_file.exists() and token_file.exists():
            print(f"Files for id={idx} already exist, skipping...")
            continue
        pending_ids.append(idx)
        pending_trunc_years.append(trunc_year)
        output_paths[idx] = (index_file, token_file)

    if not pending_ids:
        print("All requested IDs have already been generated.")
        return

    configured_samples = cfg.generate.dataloader.get("samples_per_id")
    samples_per_id = (
        int(configured_samples)
        if configured_samples is not None
        else int(cfg.generate.dataloader.reps * cfg.datamodule.batch_size)
    )
    if samples_per_id < 1:
        raise ValueError("samples_per_id must be at least 1")

    print(
        f"Generating {samples_per_id} sequence(s) for {len(pending_ids)} IDs "
        f"in batches of up to {cfg.datamodule.batch_size}."
    )
    dataloader = data.multi_idx_dataloader(
        idxs=pending_ids,
        trunc_years=pending_trunc_years,
        reps=samples_per_id,
        split=cfg.generate.dataloader.split,
    )

    trunc_year_by_id = dict(zip(pending_ids, pending_trunc_years))
    results = {
        idx: {"original_sequence": None, "generated_rows": []}
        for idx in pending_ids
    }

    model.to(cfg.trainer.accelerator)
    model.eval()
    started_ids = set()
    with tqdm(
        total=len(dataloader.dataset),
        desc="Generating sequences",
        unit="seq",
        dynamic_ncols=True,
    ) as progress:
        for batch in dataloader:
            batch_ids = [int(idx) for idx in batch["sequence_id"].tolist()]
            for idx in batch_ids:
                if idx not in started_ids:
                    progress.write(
                        f"Starting id={idx} w/o {trunc_year_by_id[idx]} years..."
                    )
                    started_ids.add(idx)

            known_masks = batch["padding_mask"].detach().clone().cpu().bool().numpy()
            original_sequences = batch["original_sequence"].detach().cpu().numpy()

            if cfg.generate.sampler.num_years is None:
                generation_years = [trunc_year_by_id[idx] for idx in batch_ids]
            else:
                generation_years = int(cfg.generate.sampler.num_years)

            batch = model.transfer_batch_to_device(
                batch, model.device, dataloader_idx=0
            )
            sample_batch, _ = model.beam_search(
                batch,
                num_years=generation_years,
                beam_width=cfg.generate.sampler.beam_width,
                length_penalty=cfg.generate.sampler.length_penalty,
                verbose=cfg.generate.sampler.verbose,
                eoy_idx=eoy_idx,
            )
            generated = sample_batch["input_ids"][:, 0].detach().cpu().numpy()

            for row, idx in enumerate(batch_ids):
                result = results[idx]
                if result["original_sequence"] is None:
                    result["original_sequence"] = original_sequences[row]
                    if not result["generated_rows"]:
                        generated[row, known_masks[row]] = 0
                result["generated_rows"].append(generated[row].copy())
            progress.update(len(batch_ids))

    for idx, result in results.items():
        if len(result["generated_rows"]) != samples_per_id:
            raise RuntimeError(
                f"Expected {samples_per_id} generations for id={idx}, got "
                f"{len(result['generated_rows'])}"
            )
        save_result(
            *output_paths[idx],
            result["original_sequence"],
            result["generated_rows"],
            index2token,
        )
"""

if __name__ == "__main__":
    main()
