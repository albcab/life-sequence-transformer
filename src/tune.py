import os
import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything, Trainer
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
from ray.tune.schedulers import ASHAScheduler

import ray
from ray import air, tune
from ray.tune import CLIReporter
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.optuna import OptunaSearch
from pathlib import Path
import logging

#need to be even
MIN_HIDDEN_SIZE, MAX_HIDDEN_SIZE = 64, 512
MIN_FF_SIZE, MAX_FF_SIZE = 128, 512
MIN_ENCODER_LAYERS, MAX_ENCODER_LAYERS = 0, 4
MIN_DECODER_LAYERS, MAX_DECODER_LAYERS = 4, 12
#need to be even
MAX_ENCODER_HEADS = 6
MAX_DECODER_HEADS = 12

STARTING_CONFIG = {
    "hidden_size": 64,
    "intermediate_size": 128,
    "encoder_layers": 0,
    "encoder_attention_heads": 2,
    "decoder_layers": 4,
    "decoder_attention_heads": 8,
}

# HOME_PATH = str(Path())
# HOME_PATH = str(Path.home() / "usr/src/l2v")
HOME_PATH = "/usr/src/l2v"

log = logging.getLogger(__name__)


def last_ckpt(dir_):
    ckpt_path = Path(HOME_PATH, dir_, "last.ckpt")
    if ckpt_path.exists():
        log.info("Checkpoint exists:\n\t%s" %str(ckpt_path))
        return None
    else:
        log.info("Checkpoint DOES NOT exists:\n\t%s" %str(ckpt_path))
        return None

def home_path():
    return HOME_PATH

try:
    OmegaConf.register_new_resolver("last_ckpt", last_ckpt)
except Exception as e:
    print(e)

"""
For learning rate: Convergence metrics like AUL or training loss might be emphasized.
For weight decay: Metrics sensitive to overfitting/underfitting (e.g., validation accuracy or loss) are important.
For batch size: Metrics like throughput or speed per training iteration could also be considered.
"""

def tune_hyperparameters(config, cfg, data):

    os.chdir(HOME_PATH)

    print("I AM HERE")
    cfg.model.hparams.hidden_size = config["hidden_size"]
    cfg.model.hparams.intermediate_size = config["intermediate_size"]
    cfg.model.hparams.encoder_layers = config["encoder_layers"]
    cfg.model.hparams.encoder_attention_heads = config["encoder_attention_heads"]
    cfg.model.hparams.decoder_layers = config["decoder_layers"]
    cfg.model.hparams.decoder_attention_heads = config["decoder_attention_heads"]
    #cfg.model.hparams.learning_rate = config["learning_rate"]
    #cfg.model.hparams.weight_decay = config["weight_decay"]

    model = instantiate(cfg.model, _convert_="all")
    tune_callback = TuneReportCheckpointCallback({"loss": "val/loss",
                                                  "perplexity": "val/perplexity",
                                                  "f1": "val/f1"}, 
                                                  on="validation_end")

    print("ALL good here")
    ##TRAINER
    trainer = Trainer(callbacks=[tune_callback], 
                      accelerator=cfg.trainer['accelerator'], 
                      devices=cfg.trainer['devices'],
                      #default_root_dir = cfg.trainer["default_root_dir"],
                      max_epochs = 6,
                      accumulate_grad_batches=cfg.trainer['accumulate_grad_batches'],
                    #   log_every_n_steps = 1,
                      #num_sanity_val_steps = 10,
                      limit_train_batches = 3750,
                      limit_val_batches = 750,
                      #limit_test_batches = 12500,
                      #check_val_every_n_epoch=1
                     )
    print("STARTED")
    ##TRAINING
    trainer.fit(model, data)

def define_by_run_func(trial):
    hidden_size = trial.suggest_int("hidden_size", MIN_HIDDEN_SIZE, MAX_HIDDEN_SIZE, step=2)
    trial.suggest_int("encoder_attention_heads", 2, MAX_ENCODER_HEADS, step=2)
    # trial.suggest_categorical("encoder_attention_heads",
    #     [i for i in range(2, MAX_ENCODER_HEADS + 1) if hidden_size % i == 0])
    trial.suggest_int("decoder_attention_heads", 2, MAX_DECODER_HEADS, step=2)
    # trial.suggest_categorical("decoder_attention_heads",
    #     [i for i in range(2, MAX_DECODER_HEADS + 1) if hidden_size % i == 0])
    trial.suggest_int("intermediate_size", MIN_FF_SIZE, MAX_FF_SIZE, step=16)
    trial.suggest_int("encoder_layers", MIN_ENCODER_LAYERS, MAX_ENCODER_LAYERS)
    trial.suggest_int("decoder_layers", MIN_DECODER_LAYERS, MAX_DECODER_LAYERS)
    return {}


import sys
@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg):

    # Workaround for hydra breaking import of local package - don't need if running as module "-m src.train"
    # sys.path.append(hydra.utils.get_original_cwd())
    
    ##GLOBAL SEED
    seed_everything(cfg.seed)
    ray.init(num_cpus=4, num_gpus=1)
    ray_config = {"hidden_size": tune.qrandint(MIN_HIDDEN_SIZE, MAX_HIDDEN_SIZE, 2),
                  "encoder_attention_heads": tune.sample_from(
                      lambda spec: tune.choice([i for i in range(2, MAX_ENCODER_HEADS + 1) if spec.config.hidden_size % i == 0])
                  ),
                  "decoder_attention_heads": tune.sample_from(
                      lambda spec: tune.choice([i for i in range(2, MAX_DECODER_HEADS + 1) if spec.config.hidden_size % i == 0])
                  ),
                  "intermediate_size": tune.qrandint(MIN_FF_SIZE, MAX_FF_SIZE, 16),
                  "encoder_layers": tune.randint(MIN_ENCODER_LAYERS, MAX_ENCODER_LAYERS), #   "encoder_attention_heads": tune.choice([2, 4, 8, 16]),
                  "decoder_layers": tune.randint(MIN_DECODER_LAYERS, MAX_DECODER_LAYERS), #   "decoder_attention_heads": tune.choice([4, 8, 12, 16]),

                #   "learning_rate": tune.loguniform(1e-4, 1e-2), #or tune.loguniform(1e-5, 5e-4) for finetuning
                #   "weight_decay": tune.loguniform(1e-5, 1e-1)
                 }
    
    data = instantiate(cfg.datamodule, _convert_="all")
    cfg.model.hparams.vocab_size = data.vocabulary.size()
    cfg.model.hparams.n_users = data.corpus.population.data_split().train.shape[0]
    
    reporter = CLIReporter(
            parameter_columns=list(ray_config.keys()),
            metric_columns=["perplexity", "f1", "loss", "training_iteration"],
            max_report_frequency=100)

    train_fn_with_parameters = tune.with_parameters(tune_hyperparameters,
                                                    cfg=cfg, data=data)

    scheduler = ASHAScheduler(max_t=6,
                             grace_period=1,
                             metric="perplexity",
                             mode="min",
                             reduction_factor=2)
    search = ConcurrencyLimiter(OptunaSearch(space=define_by_run_func, 
                                             metric="perplexity", mode="min", 
                                             points_to_evaluate=[STARTING_CONFIG]), 1)
    tuner = tune.Tuner(tune.with_resources(train_fn_with_parameters, {'cpu': 4, 'gpu': 1}),
            tune_config=tune.TuneConfig(search_alg=search,
                                    scheduler=scheduler,
                                    num_samples=100),
            run_config=air.RunConfig(name=cfg.name, progress_reporter=reporter, storage_path=HOME_PATH + "/ray/"),
            # param_space=ray_config
    )

    result = tuner.fit()


if __name__ == "__main__":
    main()