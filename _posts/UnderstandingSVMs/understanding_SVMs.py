import numpy as np
import matplotlib.pyplot as plt

b = np.array([1, 2])
b0 = 1
x = np.linspace(-5, 5, 100)
def hyperplane(x, b, b0):
    return (-b0 - b[0] * x) / b[1]

# Separating hyperplanes
plt.plot(x, hyperplane(x, b, b0), 'k-')
plt.plot(x, hyperplane(x, b, b0 + 1), 'k--')
plt.plot(x, hyperplane(x, b, b0 - 1), 'k--')
plt.legend(['Hyperplane', 'Margin', 'Margin'])
plt.axhline(0)
plt.axvline(0)
plt.savefig('assets/images/SVMs/svm_margin.png')
plt.close("all")

# Normal to the hyperplane
plt.plot(x, hyperplane(x, b, b0), c = "blue")
plt.axhline(0, c = "black", linestyle = "--")
plt.axvline(0, c = "black", linestyle = "--")
# For any vector in x
x_1 = np.array([1, hyperplane(1, b, b0)])
x_2 = np.array([3, hyperplane(3, b, b0)])
plt.scatter([x_1[0], x_2[0]], [x_1[1], x_2[1]], c = "red")
plt.quiver(*x_1, x_2[0] - x_1[0], x_2[1] - x_1[1], scale = 10, angles = "xy")
#plt.quiver(*[0, 0], x_2[0] - x_1[0], x_2[1] - x_1[1], scale = 10, angles = "xy")
# Some issue is happening with the plotting... I do not get why
# here only works with uv and not with xy....
#plt.quiver(*[0, 0], b[0], b[1], scale = 10, angles = "uv")
#plt.quiver(*[0, -3], b[0], b[1], scale = 4, angles = "uv")

plt.savefig("assets/images/SVMs/svm_normal")

# Visualization of the margin as a function of the norm of the normal
plt.close("all")
colors = ["blue", "red", "green"]
# Plot 3 subplots with a different margin
plt.figure(figsize = (15, 5))
for i, pm in enumerate([1, 2, 3]):
    plt.subplot(1, 3, i + 1)
    b_t = b * pm
    plt.plot(x, hyperplane(x, b_t, b0), c = colors[i])
    plt.plot(x, hyperplane(x, b_t, b0 + 1), c = colors[i], linestyle = "--")
    plt.plot(x, hyperplane(x, b_t, b0 - 1), c = colors[i], linestyle = "--")
    plt.axhline(0, c = "black", linestyle = "--")
    plt.axvline(0, c = "black", linestyle = "--")
    plt.title(f"Margin: {2/np.linalg.norm(b_t):.3f} \n beta: {b_t}")
plt.tight_layout()
plt.savefig("assets/images/SVMs/svm_margin_norm.png")


#### LAGRANGE MULTIPLIERS ####
def circle(x, y):
    return x**2 + y**2

def dcircle(x, y):
    return np.array([2*x, 2*y])

#  Draw unit circle in 2d
plt.close("all")
fig, ax = plt.subplots()
ax.set_xlim((-2, 2))
ax.set_ylim((-2, 2))
ax.set_aspect("equal")
ax.grid()
x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)
Z = circle(X, Y)
ax.contour(X, Y, Z, levels = [1])
# Draw gradient
x_0 = np.array([1, 1])
ax.quiver(*x_0, *dcircle(*x_0), scale = 10, angles = "xy")

plt.show()
