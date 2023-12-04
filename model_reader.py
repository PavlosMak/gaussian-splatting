import sys
from argparse import ArgumentParser

import torch
import numpy as np

from arguments import ModelParams, PipelineParams, OptimizationParams
from scene import GaussianModel


def reader(dataset, opt, path_to_checkpoint: str):
    gaussians = GaussianModel(dataset.sh_degree)
    gaussians.training_setup(opt)

    (model_params, first_iter) = torch.load(path_to_checkpoint)
    gaussians.restore(model_params, opt)
    return gaussians


