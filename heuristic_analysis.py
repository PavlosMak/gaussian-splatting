import sys
from argparse import ArgumentParser

import torch
import numpy as np

from arguments import ModelParams, PipelineParams, OptimizationParams
from model_reader import reader

import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Model reading parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])

    xs = [50, 100, 200, 500, 750, 1000, 1500, 2000, 2500, 3500]

    # generated running the metrics.py script
    ssims = [0.7037066, 0.7305199, 0.7553717, 0.758409, 0.7565377, 0.7576197, 0.7567272, 0.7551782, 0.7516125,
             0.7556071]
    psnrs = [14.5208893, 15.684171, 16.7328320, 16.9944706, 16.9567375, 17.0694828, 17.1067562, 17.0300884, 16.8142433,
             16.9733486]
    lpips = [0.3661898, 0.3387794, 0.2996052, 0.2922340, 0.2942639, 0.2928488, 0.2909563, 0.2945123, 0.2991107,
             0.2979087]

    runs = [
        f"/home/pavlos/Desktop/stuff/Uni-Masters/Q5/GraphicsSeminar/gaussian-splatting/output/heuristic_eval/{x}/chkpnt3000.pth"
        for x in xs]

    numbers = []

    for run in runs:
        print(f"Reading: {run}")
        gaussians = reader(lp.extract(args), op.extract(args), run)
        number_of_gaussians, _ = gaussians.get_xyz.shape

        numbers.append(number_of_gaussians)

    np.savetxt("numbers_heuristics.txt", numbers)

    initial_count = 12612

    plt.plot(xs, np.array(numbers) - initial_count, marker="o", markersize=3)
    plt.title("Kernel Increase over Densification Intervals for 3000 Iterations")
    plt.xlim(10)
    plt.ylim(0)
    plt.ylabel("Kernel Increase")
    plt.xlabel("Densification Intervals")
    plt.savefig("densification_intervals.pdf")
    # plt.show()
    plt.close()

    plt.plot(xs, psnrs, marker="o", markersize=3)
    plt.title("PSNR over Densification Intervals for 3000 Iterations")
    plt.xlim(10)
    plt.ylim(0)
    plt.ylabel("PSNR")
    plt.xlabel("Densification Intervals")
    plt.savefig("psnr.pdf")
    plt.show()
    plt.close()

    # All done
    print("\nReading complete.")
