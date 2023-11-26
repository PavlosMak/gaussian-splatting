import numpy as np
import matplotlib.pyplot as plt


opacities = np.loadtxt("opacities_smaller_new.txt")
numbers = np.loadtxt("numbers_smaller_new.txt")

xs = list(range(10, 1010, 10))

plt.plot(xs, numbers, marker="o", markersize=3)
plt.title("Number of Gaussian Kernels over Iterations")
plt.xlim(10)
plt.ylim(0)
plt.ylabel("Number of Kernels")
plt.xlabel("Iterations")
plt.savefig("new_smaller_kernels_over_iters.pdf")
plt.close()

plt.plot(xs, opacities, marker="o", markersize=3)
plt.title('Mean $\\alpha$ over Iterations')
plt.xlim(0)
plt.ylim(0)
plt.xlabel("Iterations")
plt.ylabel("$\\alpha$")
plt.savefig("new_smaller_alpha_over_iters.pdf")
plt.close()