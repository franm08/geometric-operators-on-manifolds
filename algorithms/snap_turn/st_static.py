import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# -------------------------------------------------------
# Output
# -------------------------------------------------------
OUTPUT_DIR = "algorithms/snap_turn/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------------------------------
# Math Helpers
# -------------------------------------------------------
def normalize(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v if n == 0 else v / n

def quat_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def rotate_vec_quat(v, q):
    v_q = np.array([0.0, *v])
    q_conj = np.array([q[0], -q[1], -q[2], -q[3]])
    return quat_mult(quat_mult(q, v_q), q_conj)[1:]

def quat_slerp(q0, q1, t):
    dot = np.dot(q0, q1)
    if dot < 0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        return normalize((1 - t) * q0 + t * q1)
    theta_0 = np.arccos(dot)
    return (
        np.sin((1 - t) * theta_0) / np.sin(theta_0) * q0 +
        np.sin(t * theta_0) / np.sin(theta_0) * q1
    )

# -------------------------------------------------------
# Data
# -------------------------------------------------------
a = normalize(np.array([1.0, 0.2, 0.1]))
b = normalize(np.array([0.1, 0.9, 0.3]))

d = np.dot(a, b)
c = np.cross(a, b)

q_raw = np.concatenate(([1 + d], c))
q = normalize(q_raw)
q_id = np.array([1.0, 0.0, 0.0, 0.0])

# SLERP samples (minimal rotation)
ts = np.linspace(0, 1, 50)
path = np.array([rotate_vec_quat(a, quat_slerp(q_id, q, t)) for t in ts])

# -------------------------------------------------------
# Plot
# -------------------------------------------------------
fig = plt.figure(figsize=(6, 6), dpi=200)
ax = fig.add_subplot(111, projection="3d")
ax.set_axis_off()
ax.set_box_aspect([1, 1, 1])

# Sphere wireframe
u = np.linspace(0, 2*np.pi, 60)
v = np.linspace(0, np.pi, 30)
x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones_like(u), np.cos(v))
ax.plot_wireframe(x, y, z, color="gray", alpha=0.25, linewidth=0.4)

# Path (great-circle arc)
ax.plot(path[:,0], path[:,1], path[:,2],
        color="green", linewidth=2, label="Minimal Rotation Path")

# Start / End vectors
ax.scatter(*a, color="navy", s=40, label=r"$\mathbf{a}$")
ax.scatter(*b, color="crimson", s=40, label=r"$\mathbf{b}$")

# Rotation axis
axis = normalize(c)
ax.plot([-axis[0], axis[0]],
        [-axis[1], axis[1]],
        [-axis[2], axis[2]],
        color="purple", linestyle="--", linewidth=1.5,
        label=r"Axis $\mathbf{a}\times\mathbf{b}$")

ax.legend(loc="upper right", fontsize=8)
ax.set_title("SnapTurn: Minimal Rotation on the Unit Sphere", fontsize=11)

# Save
out_path = os.path.join(OUTPUT_DIR, "snapturn_static.png")
plt.savefig(out_path, bbox_inches="tight")
plt.show()

print(f"Saved static SnapTurn figure to {out_path}")