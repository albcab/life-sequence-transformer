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

    nb_particles = int(cfg.generate.sampler.nb_particles)
    if nb_particles < 1:
        raise ValueError("nb_particles must be positive")
    if cfg.generate.dataloader.reps != 1:
        raise ValueError("Use nb_particles for SMC; dataloader.reps must be 1")

    pending_ids, pending_years = [], []
    output_paths = {}
    for idx, years in zip(ids, trunc_years):
        paths = tuple(dir_name / f"{idx}_{kind}.csv" for kind in ("index", "token", "weights"))
        if all(path.exists() for path in paths):
            tqdm.write(f"Files for id={idx} already exist, skipping...")
            continue
        pending_ids.append(idx)
        pending_years.append(years)
        output_paths[idx] = paths
    if not pending_ids:
        print("All requested IDs have already been generated.")
        return

    # Load each person once; the sampler creates their particle population.
    dataloader = data.multi_idx_dataloader(
        idxs=pending_ids, reps=1, trunc_years=pending_years,
        split=cfg.generate.dataloader.split,
    )
    years_by_id = dict(zip(pending_ids, pending_years))
    model.to(cfg.trainer.accelerator)
    model.eval()
    print(f"Batching up to {cfg.datamodule.batch_size} people, {nb_particles} particles/person.")
    with tqdm(total=len(pending_ids), desc="Generating users", unit="user") as progress:
        for batch_number, batch in enumerate(dataloader, start=1):
            batch_ids = [int(idx) for idx in batch["sequence_id"].tolist()]
            originals = batch["original_sequence"].detach().cpu().numpy()
            known = batch["padding_mask"].detach().cpu().bool().numpy()
            generation_years = cfg.generate.sampler.num_years
            if generation_years is None:
                generation_years = [years_by_id[idx] for idx in batch_ids]
            progress.write(f"Starting batch {batch_number}/{len(dataloader)}: IDs {batch_ids}")
            batch = model.transfer_batch_to_device(batch, model.device, dataloader_idx=0)
            sample_batch, _, final_weights, _ = model.smc_sample(
                batch, num_years=generation_years, nb_particles=nb_particles,
                ess_threshold=cfg.generate.sampler.ess_threshold,
                verbose=cfg.generate.sampler.verbose,
                eoy_idx=eoy_idx, ending_idx=eol_idx,
            )
            rows = sample_batch["input_ids"][:, 0].detach().cpu().numpy()
            rows = rows.reshape(len(batch_ids), nb_particles, -1)
            weights = final_weights.detach().cpu().numpy().reshape(len(batch_ids), nb_particles)
            for row, idx in enumerate(batch_ids):
                rows[row][:, known[row]] = 0
                index_file, token_file, weight_file = output_paths[idx]
                save_result(index_file, token_file, originals[row], list(rows[row]), index2token)
                np.savetxt(weight_file, weights[row], fmt="%.10g")
            progress.set_postfix(batch=f"{batch_number}/{len(dataloader)}", particles=nb_particles)
            progress.update(len(batch_ids))

if __name__ == "__main__":
    main()
