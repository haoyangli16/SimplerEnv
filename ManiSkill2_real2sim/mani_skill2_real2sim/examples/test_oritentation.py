import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_orientation(ax, rotation, center=[0, 0, 0], scale=1):
    """Plot the orientation axes given a rotation."""
    R = rotation.as_matrix()
    colors = ["r", "g", "b"]
    for i in range(3):
        ax.quiver(
            center[0],
            center[1],
            center[2],
            R[0, i] * scale,
            R[1, i] * scale,
            R[2, i] * scale,
            color=colors[i],
            arrow_length_ratio=0.1,
        )


# Initial and target quaternions
initial_quat = [1, 0, 0, 0]  # Identity quaternion (no rotation)
target_quat = [0, -0.707, -0.707, 0]

# Create key rotations and times
key_rots = R.from_quat([initial_quat, target_quat])
key_times = [0, 1]  # Normalize time to 0-1 range

# Create the Slerp interpolator
slerp = Slerp(key_times, key_rots)

# Create figure and 3D axis
fig = plt.figure(figsize=(15, 5))
n_steps = 5
for step in range(n_steps):
    ax = fig.add_subplot(1, n_steps, step + 1, projection="3d")

    # Interpolate rotation
    t = step / (n_steps - 1)
    current_rot = slerp([t])[
        0
    ]  # Slerp returns an array, we want the first (and only) element

    # Plot the orientation
    plot_orientation(ax, current_rot)

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Step {step+1}")

plt.tight_layout()
plt.show()

# Calculate and print the required rotation
initial_rot = R.from_quat(initial_quat)
target_rot = R.from_quat(target_quat)

diff_rot = target_rot * initial_rot.inv()
diff_euler = diff_rot.as_euler("xyz", degrees=True)

print("Required rotation (in degrees):")
print(f"Roll (around x-axis): {diff_euler[0]:.2f}")
print(f"Pitch (around y-axis): {diff_euler[1]:.2f}")
print(f"Yaw (around z-axis): {diff_euler[2]:.2f}")
