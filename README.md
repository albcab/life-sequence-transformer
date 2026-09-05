# Life Sequence Transformer: Generative Modelling for Counterfactual Simulation

Code for model architecture using [pytorch](https://github.com/pytorch/pytorch), experiments run with [pytorch-lightning](https://github.com/Lightning-AI/pytorch-lightning) and [hydra](https://github.com/facebookresearch/hydra) for configuring hyperparameters.
Database management using [dask](https://github.com/dask/dask), hyperparameter optimization using [optuna](https://github.com/optuna/optuna) and [ray](https://github.com/ray-project/ray), performer implementation based on [perfomer-pytorch](https://github.com/lucidrains/performer-pytorch) using [fast-transformers](https://github.com/idiap/fast-transformers) CUDA builds.

This codebase is based on [life2vec](https://github.com/SocialComplexityLab/life2vec) from the paper [Using Sequences of Life-events to Predict Human Lives](https://www.nature.com/articles/s43588-023-00573-5).

### Overall Structure

The `/conf` folder contains configs for the experiments:
1. `/experiment` contains configuration for training.
2. `/tasks` contain configuration for data augmentation.
3. `/trainer` and `/datamodule` contain configuration for lightning's `Trainer`.
4. `/data_new` contains configuration for data loading and processing.
5. `callbacks.yaml` contains configuration for the lightning's `Callbacks`.
6. `prepare_data.yaml` can be used to run data preprocessing.

The `/src` folder contains the source code:
1. The `/src/dataloaders` contains scripts to preprocess, augment and load data.
2. The `/src/models` contains the model's source code.
3. `train.py`, `finetune.py`, `test.py`, `tune.py` are used to run a particular stage of the training.
4. `prepare_data.py` was used to run the data processing.
5. `sample_idx.py` and `multiple_idx.py` are used to generate sequences for individuals in the database, conditioned on some know years.

If using NVIDIA GPUs, you can build a container using a local `Dockerfile`.
Dockerfiles, CSV datasets, generated results, caches, and model artifacts are
ignored by Git and must be supplied or generated locally.

### Run Training and Experiments

```
# build datasets
HYDRA_FULL_ERROR=1 python -m src.prepare_data experiment=decode_only

# run training
HYDRA_FULL_ERROR=1 python -m src.train experiment=decode_only

# run finetuning
HYDRA_FULL_ERROR=1 python -m src.finetune generate=decode_only

# run sequence generation (requires specifying parameters)
HYDRA_FULL_ERROR=1 python -m src.multiple_idx generate=decode_only datamodule.batch_size=8 generate.dataloader.file_name=...
```
