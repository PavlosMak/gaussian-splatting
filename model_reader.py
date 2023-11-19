import sys
from argparse import ArgumentParser

import torch

from arguments import ModelParams, PipelineParams, OptimizationParams
from scene import GaussianModel


def reader(dataset, opt, path_to_checkpoint: str):
    gaussians = GaussianModel(dataset.sh_degree)
    gaussians.training_setup(opt)

    (model_params, first_iter) = torch.load(path_to_checkpoint)
    gaussians.restore(model_params, opt)
    return gaussians


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Model reading parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])

    print("Reading: " + args.checkpoint_path)
    gaussians = reader(lp.extract(args), op.extract(args), args.checkpoint_path)

    print(gaussians)
    # All done
    print("\nReading complete.")
