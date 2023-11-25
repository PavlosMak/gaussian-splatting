import numpy as np
import matplotlib.pyplot as plt


opacities = np.loadtxt("opacities.txt")
numbers = np.loadtxt("numbers.txt")

xs = list(range(10, 4010, 10))

plt.plot(xs, numbers)
plt.title("Number of Gaussian Kernels over Iterations")
plt.xlim(10)
plt.ylim(0)
plt.ylabel("Number of Kernels")
plt.xlabel("Iterations")
plt.savefig("kernels_over_iters.pdf")
plt.close()

plt.plot(xs, opacities)
plt.title('Mean $\\alpha$ over Iterations')
plt.xlim(0)
plt.ylim(0)
plt.xlabel("Iterations")
plt.ylabel("$\\alpha$")
plt.savefig("alpha_over_iters.pdf")
plt.close()