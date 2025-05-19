import re
import hydra
from hydra.utils import instantiate, get_class
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything, Trainer
import os
from pathlib import Path
import logging
import numpy as np
import pandas as pd

HOME_PATH = str(Path())

log = logging.getLogger(__name__)

def last_ckpt(dir_):
    print(HOME_PATH)
    ckpt_path = Path(HOME_PATH, dir_, "best.ckpt")
    ckpt_ = Path(HOME_PATH, dir_)
    if ckpt_path.exists():
        print("CHECKPOINT EXISTS:", str(ckpt_path))
        return str(ckpt_path)
    elif ckpt_.exists():
        print("CHECKPOINT EXISTS:", str(ckpt_))
        return str(ckpt_)
    else:
        print("CHECKPOINT DOES NOT EXISTS:", str(ckpt_path))
        raise Exception
        return None


@hydra.main(config_path="../conf", config_name="gconfig", version_base=None)
def main(cfg):

    # Workaround for hydra breaking import of local package - don't need if running as module "-m src.train"
    # sys.path.append(hydra.utils.get_original_cwd())
    # print(cfg.ckpt_path)
    
    ##GLOBAL SEED
    seed_everything(cfg.seed)
    
    data = instantiate(cfg.datamodule, _convert_="all")
    cfg.model.hparams.vocab_size = data.vocabulary.size()
    print(cfg.model.hparams.vocab_size)
    cfg.model.hparams.n_users = data.corpus.population.data_split().train.shape[0]

    ##MODEL
    ModelClass = get_class(cfg.model._target_)

    if cfg.get("model_path"):
        model = ModelClass.load_from_checkpoint(
            cfg.model_path,
            # last_ckpt(cfg.ckpt_path),
            strict=False,
            hparams=cfg.model.hparams,
        )
    else:
        raise NotImplementedError("No pretrained model checkpoint to load")
    
    dir_name = f"generated/{cfg.implementation}/{cfg.name}/{cfg.generate.dataloader.file_name.split(".")[0]}/"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    token2index = data.vocabulary.token2index
    index2token = data.vocabulary.index2token
    eoy_idx = token2index['[EOY]']
    ending_idx = token2index['TIPO_10']
    model.to(cfg.trainer.accelerator)
    model.eval()
    ids_df = pd.read_csv(cfg.generate.dataloader.file_name)
    idxs = ids_df.USER_ID.tolist()
    trunc_years = ids_df.year_to_remove.to_list()

    for idx, trunc_year in zip(idxs, trunc_years):

        print()
        print(f"Starting id={idx} w/o {trunc_year} years...")
        dataloader = data.single_idx_dataloader(
            idx=idx,
            trunc_years=trunc_year,
            reps=cfg.generate.dataloader.reps,
            split=cfg.generate.dataloader.split)

        name = None
        cum_rows = []
        for batch in dataloader:

            if name is None:
                name = batch['sequence_id'][0].item()
                original_sequence = batch['original_sequence'][0].detach().cpu().numpy()
                known = batch['padding_mask'][0].detach().clone().cpu().bool()

            batch = model.transfer_batch_to_device(batch, model.device, dataloader_idx=0)
            
            sample_batch, _ = model.sample(
                batch,
                num_years=cfg.generate.sampler.num_years or trunc_year,
                temp=cfg.generate.sampler.temp,
                verbose=cfg.generate.sampler.verbose,
                eoy_idx=eoy_idx,
                # ending_idx=ending_idx,
            )
            rows = sample_batch['input_ids'][:, 0].detach().cpu().numpy()
            rows[0, known] = 0
            cum_rows.append(rows)

        data_rows = np.vstack([original_sequence] + cum_rows)
        np.savetxt(dir_name + f"{name}_index.csv", data_rows, fmt="%d", delimiter=",")
        np.savetxt(dir_name + f"{name}_token.csv", np.vectorize(lambda i: index2token[i])(data_rows), fmt="%s", delimiter=",")

if __name__ == "__main__":
    main()