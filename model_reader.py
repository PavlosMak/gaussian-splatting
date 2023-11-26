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


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Model reading parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])

    # checkpoints = [f"/media/pavlos/One Touch/unity_train_checkpoints/chkpnt{i}.pth" for i in range(10, 4010, 10)]
    checkpoints = [f"/home/pavlos/Desktop/stuff/Uni-Masters/Q5/GraphicsSeminar/gaussian-splatting/output/train_smaller_new/chkpnt{i}.pth" for i in range(10, 1010, 10)]

    opacities = []
    numbers = []

    # gaussians = reader(lp.extract(args), op.extract(args), args.checkpoint_path)
    for checkpoint in checkpoints:
        print(f"Reading: {checkpoint}")
        gaussians = reader(lp.extract(args), op.extract(args), checkpoint)
        mean_opacity = torch.mean(gaussians.get_opacity)
        number_of_gaussians, _ = gaussians.get_xyz.shape

        opacities.append(mean_opacity.cpu().detach())
        numbers.append(number_of_gaussians)

    np.savetxt("opacities_smaller_new.txt", opacities)
    np.savetxt("numbers_smaller_new.txt", numbers)

    # All done
    print("\nReading complete.")
