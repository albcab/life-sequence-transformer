# HYDRA_FULL_ERROR=1 python -m src.train experiment=pretrain trainer.devices=[1]
# HYDRA_FULL_ERROR=1 python -m src.prepare_data experiment=pretrain

HYDRA_FULL_ERROR=1 python -m src.train experiment=decode trainer.devices=[1]
HYDRA_FULL_ERROR=1 python -m src.prepare_data experiment=decode
