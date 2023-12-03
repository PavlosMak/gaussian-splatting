import sys
from argparse import ArgumentParser

import torch
import pandas as pd

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

    output = "/home/pavlos/Desktop/stuff/Uni-Masters/Q5/GraphicsSeminar/train_smaller_frames_3000"

    for checkpoint in range(1010, 3010, 10):
        checkpoint_path = f"/home/pavlos/Desktop/stuff/Uni-Masters/Q5/GraphicsSeminar/gaussian-splatting/output/train_smaller_3000_500/chkpnt{checkpoint}.pth"
        print(f"Reading: {checkpoint}")
        gaussians = reader(lp.extract(args), op.extract(args), checkpoint_path)
        mean_opacity = torch.mean(gaussians.get_opacity)
        number_of_gaussians, _ = gaussians.get_xyz.shape
        entries = []
        for i in range(number_of_gaussians):
            position = gaussians.get_xyz[i]
            opacity = gaussians.get_opacity[i]
            rotation = gaussians.get_rotation[i]
            scale = gaussians.get_scaling[i]
            rgb = gaussians.get_features[i][0]
            row = {
                "id": i,
                "position_x": position[0].item(),
                "position_y": position[1].item(),
                "position_z": position[2].item(),
                "scale_x": scale[0].item(),
                "scale_y": scale[1].item(),
                "scale_z": scale[2].item(),
                "rot_x": rotation[0].item(),
                "rot_y": rotation[1].item(),
                "rot_z": rotation[2].item(),
                "rot_w": rotation[3].item(),
                "opacity": opacity.item(),
                "red_feature": rgb[0].item(),
                "green_feature": rgb[1].item(),
                "blue_feature": rgb[2].item()
            }
            entries.append(row)
        frame_df = pd.DataFrame(entries)
        frame_df.to_csv(f"{output}/{checkpoint}.csv")

    # All done
    print("\nFrame creation complete.")
