import numpy as np
import cv2
import matplotlib.pyplot as plt

ground_truth = "/home/pavlos/Desktop/stuff/Uni-Masters/Q5/GraphicsSeminar/unity_train_smaller_comparisons/ground_truth"
renders = "/home/pavlos/Desktop/stuff/Uni-Masters/Q5/GraphicsSeminar/unity_train_smaller_comparisons/render"

mse_path = "/home/pavlos/Desktop/stuff/Uni-Masters/Q5/GraphicsSeminar/unity_train_smaller_comparisons/mse"

def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2, axis=2)
    return err

for i in range(1,4):
    truth_path = f"{ground_truth}/{i}.png"
    render_path = f"{renders}/{i}.png"

    truth = cv2.imread(truth_path)
    render = cv2.imread(render_path)

    error = mse(truth, render)
    plt.imshow(error)
    plt.grid(False)
    plt.axis("off")
    plt.savefig(f"{mse_path}/{i}.png")
