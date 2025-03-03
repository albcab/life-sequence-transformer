# HYDRA_FULL_ERROR=1 python -m src.train experiment=pretrain trainer.devices=[1]
# HYDRA_FULL_ERROR=1 python -m src.prepare_data experiment=pretrain

HYDRA_FULL_ERROR=1 python -m src.train experiment=decode trainer.devices=[1]
HYDRA_FULL_ERROR=1 python -m src.prepare_data experiment=decode

PL_GLOBAL_SEED=2025 HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=1 python -m src.tune_encoder experiment=pretrain trainer.devices=[0]
PL_GLOBAL_SEED=2025 HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0 python -m src.tune experiment=pretrain trainer.devices=[0]