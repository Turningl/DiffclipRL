# -*- coding: utf-8 -*-
# @Author : liang
# @File : run_ddpm.py


import hydra
import omegaconf
import os
import torch
from omegaconf import DictConfig, OmegaConf

from diffclipRL.mattergen.common.utils.globals import MODELS_PROJECT_ROOT
from diffclipRL.mattergen.diffusion.config import Config
from diffclipRL.mattergen.diffusion.run import main

import warnings
warnings.filterwarnings('ignore')


@hydra.main(
    config_path=str(MODELS_PROJECT_ROOT), config_name="default", version_base="1.1"
)
def mattergen_main(cfg: omegaconf.DictConfig):
    # Tensor Core acceleration (leads to ~2x speed-up during training)
    torch.set_float32_matmul_precision("high")
    # Make merged config options
    # CLI options take priority over YAML file options
    schema = OmegaConf.structured(Config)
    config = OmegaConf.merge(schema, cfg)
    OmegaConf.set_readonly(config, True)  # should not be written to
    print(OmegaConf.to_yaml(cfg, resolve=True))

    main(config)

if __name__ == '__main__':

    os.environ["HYDRA_FULL_ERROR"] = "1"
    # os.environ["PYTORCH_LIGHTNING_TQDM_DISABLE_SUB_BARS"] = "1"

    os.environ["WANDB_API_KEY"] = 'b609449b7b81fba5c9d5423663520f5888c7df01'
    os.environ["WANDB_MODE"] = "offline"

    mattergen_main()