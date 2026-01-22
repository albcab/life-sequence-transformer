import re
import hydra
from hydra.utils import instantiate, get_class
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything, Trainer
import sys
from pathlib import Path
import logging
import numpy as np

HOME_PATH = str(Path())

log = logging.getLogger(__name__)


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
            strict=False,
            hparams=cfg.model.hparams,
        )
    else:
        raise NotImplementedError("No pretrained model checkpoint to load")

    token2index = data.vocabulary.token2index
    index2token = data.vocabulary.index2token
    dataloader = data.single_idx_dataloader(
            idxs=cfg.generate.dataloader.idx,
            trunc_years=cfg.generate.dataloader.trunc_years,
            reps=cfg.generate.dataloader.reps * cfg.datamodule.batch_size,
            split=cfg.generate.dataloader.split)
    name = None
    eoy_idx = token2index['[EOY]']
    cum_rows = []
    model.to(cfg.trainer.accelerator)
    model.eval()
    for batch in dataloader:

        if name is None:
            name = batch['sequence_id'][0].item()
            original_sequence = batch['original_sequence'][0].detach().cpu().numpy()
            known = batch['padding_mask'][0].detach().clone().cpu().bool()

        batch = model.transfer_batch_to_device(batch, model.device, dataloader_idx=0)
        
        sample_batch, _ = model.sample(
            batch,
            num_years=cfg.generate.sampler.num_years or cfg.generate.dataloader.trunc_years,
            temp=cfg.generate.sampler.temp or 0.8,
            verbose=cfg.generate.sampler.verbose or True,
            eoy_idx=eoy_idx
        )
        rows = sample_batch['input_ids'][:, 0].detach().cpu().numpy()
        rows[0, known] = 0
        cum_rows.append(rows)

    data = np.vstack([original_sequence] + cum_rows)
    file_name = f"generated/{cfg.implementation}/{cfg.name}/{name}"
    np.savetxt(file_name + "_index.csv", data, fmt="%d", delimiter=",")
    np.savetxt(file_name + "_token.csv", np.vectorize(lambda i: index2token[i])(data), fmt="%s", delimiter=",")

if __name__ == "__main__":
    main()