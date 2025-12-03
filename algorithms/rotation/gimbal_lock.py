import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os

# -------------------------------------------------------
# 1. Output Configuration
# -------------------------------------------------------
OUTPUT_DIR = "algorithms/rotation/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
SAVE_PATH = os.path.join(OUTPUT_DIR, "gimbal_lock_anim.gif")

# -------------------------------------------------------
# 2. Math Helpers
# -------------------------------------------------------
def rotation_matrix(axis, theta):
    c, s = np.cos(theta), np.sin(theta)
    if axis == 'x': return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == 'y': return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    if axis == 'z': return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return np.eye(3)

def generate_torus_mesh(major_radius, minor_radius, axis_alignment='z', u_res=40, v_res=20):
    u = np.linspace(0, 2 * np.pi, u_res) 
    v = np.linspace(0, 2 * np.pi, v_res) 
    U, V = np.meshgrid(u, v)

    X = (major_radius + minor_radius * np.cos(V)) * np.cos(U)
    Y = (major_radius + minor_radius * np.cos(V)) * np.sin(U)
    Z = minor_radius * np.sin(V)

    if axis_alignment == 'x':
        R = rotation_matrix('y', np.pi/2)
    elif axis_alignment == 'y':
        R = rotation_matrix('x', -np.pi/2)
    else:
        return X, Y, Z

    points = np.vstack([X.flatten(), Y.flatten(), Z.flatten()])
    rotated_points = R @ points
    return rotated_points[0, :].reshape(X.shape), \
           rotated_points[1, :].reshape(X.shape), \
           rotated_points[2, :].reshape(X.shape)

def transform_mesh_data(X, Y, Z, R_matrix):
    shape = X.shape
    points = np.vstack([X.flatten(), Y.flatten(), Z.flatten()])
    rotated_points = R_matrix @ points
    return rotated_points[0, :].reshape(shape), \
           rotated_points[1, :].reshape(shape), \
           rotated_points[2, :].reshape(shape)

# -------------------------------------------------------
# 3. Setup Figure
# -------------------------------------------------------
fig = plt.figure(figsize=(10, 8), dpi=100, facecolor='white')
ax = fig.add_subplot(111, projection='3d')
ax.set_axis_off()
ax.set_box_aspect([1, 1, 1])
ax.set_facecolor('white')

LIMIT = 1.15 
ax.set_xlim(-LIMIT, LIMIT); ax.set_ylim(-LIMIT, LIMIT); ax.set_zlim(-LIMIT, LIMIT)
ax.view_init(elev=25, azim=50)

# -------------------------------------------------------
# 4. Create Geometry
# -------------------------------------------------------
TUBE_RAD = 0.06 
X_out_b, Y_out_b, Z_out_b = generate_torus_mesh(1.1, TUBE_RAD, 'x')
X_mid_b, Y_mid_b, Z_mid_b = generate_torus_mesh(0.9, TUBE_RAD, 'y')
X_inn_b, Y_inn_b, Z_inn_b = generate_torus_mesh(0.7, TUBE_RAD, 'z')

u_cone = np.linspace(0, 2*np.pi, 20)
v_cone = np.linspace(0, 1, 10)
U_c, V_c = np.meshgrid(u_cone, v_cone)
Z_c_b = 0.6 * V_c - 0.3 
X_c_b = (0.6 - Z_c_b -0.3) * 0.2 * np.cos(U_c)
Y_c_b = (0.6 - Z_c_b -0.3) * 0.2 * np.sin(U_c)

surfs = {}
props = dict(boxstyle='round', facecolor='whitesmoke', alpha=0.9, edgecolor='gray')
status_text = ax.text2D(0.05, 0.92, "", transform=ax.transAxes, fontsize=12, 
                        family='monospace', bbox=props, color='black')

# -------------------------------------------------------
# 5. Animation Loop
# -------------------------------------------------------
frames = 200
phase1_end = 90

def update(frame):
    for key in surfs:
        if surfs[key] is not None: surfs[key].remove()
    
    if frame < phase1_end:
        t_norm = frame / phase1_end
        t_smooth = t_norm * t_norm * (3 - 2 * t_norm)
        pitch = np.deg2rad(90 * t_smooth)
        roll = 0; yaw = 0
        msg = f"PHASE 1: ALIGNING AXES\nPitching Green ring to 90°...\nAngle: {np.rad2deg(pitch):.1f}°"
        status_text.set_bbox(dict(boxstyle='round', facecolor='whitesmoke', alpha=0.9, edgecolor='gray'))
    else:
        pitch = np.pi / 2 
        t_loc = (frame - phase1_end) / 40.0
        roll = np.sin(t_loc) * 0.8
        yaw  = np.cos(t_loc) * 0.8
        msg = "PHASE 2: GIMBAL LOCK!\nRed & Blue rotation axes\nare now parallel.\n(One degree of freedom lost)"
        status_text.set_bbox(dict(boxstyle='round', facecolor='#ffcccc', alpha=0.9, edgecolor='red'))

    status_text.set_text(msg)

    R_r = rotation_matrix('x', roll)
    R_p = rotation_matrix('y', pitch)
    R_y = rotation_matrix('z', yaw)

    X_o, Y_o, Z_o = transform_mesh_data(X_out_b, Y_out_b, Z_out_b, R_r)
    surfs['outer'] = ax.plot_surface(X_o, Y_o, Z_o, color='crimson', shade=True, alpha=1.0)

    R_mid_total = R_r @ R_p
    X_m, Y_m, Z_m = transform_mesh_data(X_mid_b, Y_mid_b, Z_mid_b, R_mid_total)
    surfs['mid'] = ax.plot_surface(X_m, Y_m, Z_m, color='limegreen', shade=True, alpha=0.95)

    R_inn_total = R_mid_total @ R_y
    X_i, Y_i, Z_i = transform_mesh_data(X_inn_b, Y_inn_b, Z_inn_b, R_inn_total)
    surfs['inner'] = ax.plot_surface(X_i, Y_i, Z_i, color='royalblue', shade=True, alpha=0.95)

    X_c, Y_c, Z_c = transform_mesh_data(X_c_b, Y_c_b, Z_c_b, R_inn_total)
    surfs['payload'] = ax.plot_surface(X_c, Y_c, Z_c, color='gold', shade=True, zorder=10)

    return surfs.values()

anim = FuncAnimation(fig, update, frames=frames, interval=50, blit=False)
print(f"Saving animation to {SAVE_PATH}...")
anim.save(SAVE_PATH, writer="pillow", fps=30)
print("Done.")