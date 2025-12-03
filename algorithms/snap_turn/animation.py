import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# -------------------------------------------------------
# Output Configuration
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
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    
    if dot > 0.9995:
        return normalize((1.0 - t) * q0 + t * q1)

    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * t
    sin_theta = np.sin(theta)
    
    s0 = np.sin(theta_0 - theta) / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return s0 * q0 + s1 * q1

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    if s == 0:
        return np.eye(3)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

# -------------------------------------------------------
# Data Setup
# -------------------------------------------------------
a = normalize(np.array([1.0, 0.2, 0.1]))
b = normalize(np.array([0.1, 0.9, 0.3]))

d = np.dot(a, b)
c = np.cross(a, b)
q_raw = np.concatenate(([1.0 + d], c))
q = normalize(q_raw)
q_id = np.array([1.0, 0.0, 0.0, 0.0])

num_frames = 100
ts = np.linspace(0.0, 1.0, num_frames)
ts = np.concatenate([ts, np.ones(30)]) 

rotated_vectors = []
for t in ts:
    q_t = quat_slerp(q_id, q, t)
    v_t = rotate_vec_quat(a, q_t)
    rotated_vectors.append(v_t)
rotated_vectors = np.array(rotated_vectors)

# -------------------------------------------------------
# Rocket Drawing Helper
# -------------------------------------------------------
def get_rocket_geometry(position, direction, scale=0.15):
    """
    Returns X, Y, Z arrays for a simple rocket shape oriented along 'direction'
    placed at 'position'.
    """
    # Define a simple cone pointing up Z
    r = np.linspace(0, scale/3, 10)
    theta = np.linspace(0, 2*np.pi, 20)
    R, THETA = np.meshgrid(r, theta)
    
    # Cone coordinates (local)
    X_cone = R * np.cos(THETA)
    Y_cone = R * np.sin(THETA)
    Z_cone = -np.sqrt(X_cone**2 + Y_cone**2) * 3  # Pointy end at origin? No, let's flip it
    
    # Actually let's make the cone tip at +Z
    Z_cone = (scale - np.sqrt(X_cone**2 + Y_cone**2) * 3)
    
    # Rotate points to match direction
    # Standard 'up' is [0, 0, 1]
    R_mat = rotation_matrix_from_vectors(np.array([0, 0, 1]), direction)
    
    # Apply rotation and translation
    def transform(x, y, z):
        points = np.vstack([x.flatten(), y.flatten(), z.flatten()])
        rotated = R_mat @ points
        return (rotated[0, :] + position[0]).reshape(x.shape), \
               (rotated[1, :] + position[1]).reshape(y.shape), \
               (rotated[2, :] + position[2]).reshape(z.shape)

    return transform(X_cone, Y_cone, Z_cone)

# -------------------------------------------------------
# Aesthetic Plotting
# -------------------------------------------------------
fig = plt.figure(figsize=(10, 8), dpi=120)
ax = fig.add_subplot(111, projection="3d")

ax.set_axis_off()
ax.set_box_aspect([1, 1, 1])
ax.view_init(elev=20, azim=-60)

# Sphere
u = np.linspace(0, 2 * np.pi, 60)
v = np.linspace(0, np.pi, 30)
x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_wireframe(x, y, z, color="black", alpha=0.3, linewidth=0.5)

# Equator / Meridians
theta_circ = np.linspace(0, 2*np.pi, 100)
ax.plot(np.cos(theta_circ), np.sin(theta_circ), 0, color='black', alpha=0.5, lw=1, ls='--')
ax.plot(np.zeros_like(theta_circ), np.cos(theta_circ), np.sin(theta_circ), color='silver', alpha=0.5, lw=1, ls='--')

# Rotation Axis
rot_axis = normalize(c) * 1.3
ax.plot([-rot_axis[0], rot_axis[0]], 
        [-rot_axis[1], rot_axis[1]], 
        [-rot_axis[2], rot_axis[2]], 
        color='purple', alpha=0.4, linestyle='dashdot', linewidth=1.5, label="Rotation Axis")

# Static Start/End Points
ax.scatter([a[0]], [a[1]], [a[2]], color="navy", s=50, label="Start")
ax.scatter([b[0]], [b[1]], [b[2]], color="crimson", s=50, label="Target")

# Dynamic Elements
trace_line, = ax.plot([], [], [], color="limegreen", linewidth=1.5, alpha=0.6, label="Path")
rocket_plot = ax.plot_surface(x, y, z, color='orange') # Placeholder

# Fix Legend Position (bbox_to_anchor moves it outside)
ax.legend(loc="upper right", bbox_to_anchor=(1.1, 1.0), frameon=False, fontsize=10)
ax.set_title(r"SnapTurn: Minimal Angle Rotation ($\theta$)", fontsize=14, y=0.95)

def update(frame):
    global rocket_plot
    current_vec = rotated_vectors[frame]
    
    # Update trace
    history = rotated_vectors[:frame+1]
    trace_line.set_data(history[:, 0], history[:, 1])
    trace_line.set_3d_properties(history[:, 2])
    
    # Update Rocket (remove old surface, plot new one)
    if rocket_plot:
        rocket_plot.remove()
        
    X_r, Y_r, Z_r = get_rocket_geometry(current_vec, current_vec, scale=0.2)
    rocket_plot = ax.plot_surface(X_r, Y_r, Z_r, color='orange', alpha=1.0, shade=True)
    
    return trace_line, rocket_plot

anim = FuncAnimation(fig, update, frames=len(ts), interval=20, blit=False) 
# Note: blit=False is safer for 3D surfaces in matplotlib

# Save
save_path = os.path.join(OUTPUT_DIR, "snapturn_rocket.gif")
print(f"Rendering to {save_path}...")
anim.save(save_path, writer="pillow", fps=30)
print("Done.")

plt.show()