import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os

# -------------------------------------------------------
# Config
# -------------------------------------------------------
OUTPUT_DIR = "algorithms/rotation/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
SAVE_PATH = os.path.join(OUTPUT_DIR, "rotation_matrices_anim.gif")

# -------------------------------------------------------
# Helpers
# -------------------------------------------------------
def get_cube_verts(scale=1.0):
    r = [-scale, scale]
    verts = []
    for s1 in r:
        for s2 in r:
            for s3 in r:
                verts.append([s1, s2, s3])
    verts = np.array(verts)
    
    # Front, Back, Bottom, Top, Left, Right
    faces = [
        [verts[0], verts[1], verts[3], verts[2]], 
        [verts[4], verts[5], verts[7], verts[6]], 
        [verts[0], verts[1], verts[5], verts[4]], 
        [verts[2], verts[3], verts[7], verts[6]], 
        [verts[0], verts[2], verts[6], verts[4]], 
        [verts[1], verts[3], verts[7], verts[5]]  
    ]
    return verts, faces

def rotation_matrix(axis, theta):
    c, s = np.cos(theta), np.sin(theta)
    if axis == 'x': 
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == 'y': 
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    if axis == 'z': 
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return np.eye(3)

# Helper to format numbers nicely
def fmt(val):
    if abs(val) < 1e-9: val = 0.0
    return f"{val:5.2f}"

def format_matrix_str(M):
    return (
        f"⎡ {fmt(M[0,0])}  {fmt(M[0,1])}  {fmt(M[0,2])} ⎤\n"
        f"⎢ {fmt(M[1,0])}  {fmt(M[1,1])}  {fmt(M[1,2])} ⎥\n"
        f"⎣ {fmt(M[2,0])}  {fmt(M[2,1])}  {fmt(M[2,2])} ⎦"
    )

# -------------------------------------------------------
# Plot Setup
# -------------------------------------------------------
fig = plt.figure(figsize=(14, 8), facecolor='white')

# 1. 3D Plot (Left side)
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.set_facecolor('white')
ax.set_axis_off()
ax.set_box_aspect([1, 1, 1])
LIMIT = 1.5
ax.set_xlim(-LIMIT, LIMIT); ax.set_ylim(-LIMIT, LIMIT); ax.set_zlim(-LIMIT, LIMIT)
ax.view_init(elev=20, azim=30)

# 2. Math Display (Right side) - Using relative coordinates (0-1)
ax_text = fig.add_subplot(1, 2, 2)
ax_text.axis('off')

# -------------------------------------------------------
# Static Elements setup
# -------------------------------------------------------
# Grey world axes lines
ax.plot([-2, 2], [0, 0], [0, 0], c='lightgray', lw=1, ls='--')
ax.plot([0, 0], [-2, 2], [0, 0], c='lightgray', lw=1, ls='--')
ax.plot([0, 0], [0, 0], [-2, 2], c='lightgray', lw=1, ls='--')

# Cube Data & Placeholder
v_raw, f_raw = get_cube_verts(0.5)
poly = Poly3DCollection([], alpha=0.8, edgecolor='k')
ax.add_collection3d(poly)

# Axis Arrows (Basis Vectors) placeholders
quiver_x = ax.quiver([],[],[], [],[],[])
quiver_y = ax.quiver([],[],[], [],[],[])
quiver_z = ax.quiver([],[],[], [],[],[])

# --- Text Placeholders (The 3 Matrices) ---
# Using transAxes so coordinates are 0.0 to 1.0 relative to the subplot
x_pos = 0.1
title_font = {'fontsize': 16, 'weight': 'bold'}
mat_font = {'fontsize': 14, 'family': 'monospace'}

# Rx section (Top)
t_rx = ax_text.text(x_pos, 0.80, "Rotation X ($R_x$)", **title_font, color='black')
txt_rx = ax_text.text(x_pos, 0.78, "", **mat_font, va='top')

# Ry section (Middle)
t_ry = ax_text.text(x_pos, 0.50, "Rotation Y ($R_y$)", **title_font, color='black')
txt_ry = ax_text.text(x_pos, 0.48, "", **mat_font, va='top')

# Rz section (Bottom)
t_rz = ax_text.text(x_pos, 0.20, "Rotation Z ($R_z$)", **title_font, color='black')
txt_rz = ax_text.text(x_pos, 0.18, "", **mat_font, va='top')

identity_str = format_matrix_str(np.eye(3))

# -------------------------------------------------------
# Animation
# -------------------------------------------------------
frames_per_axis = 60
total_frames = frames_per_axis * 3

def update(frame):
    global poly, quiver_x, quiver_y, quiver_z
    
    # 1. Determine active axis and angle
    if frame < frames_per_axis:
        active_axis = 'x'
        theta = 2 * np.pi * (frame / frames_per_axis)
    elif frame < frames_per_axis * 2:
        active_axis = 'y'
        theta = 2 * np.pi * ((frame - frames_per_axis) / frames_per_axis)
    else:
        active_axis = 'z'
        theta = 2 * np.pi * ((frame - frames_per_axis*2) / frames_per_axis)
        
    # 2. Calculate active rotation matrix
    R_active = rotation_matrix(active_axis, theta)
    active_mat_str = format_matrix_str(R_active)
    angle_str = f"{np.degrees(theta):.0f}°"
    
    # 3. Update 3D Scene
    # Update Cube
    v_rot = (R_active @ v_raw.T).T
    faces_rot = [[v_rot[0], v_rot[1], v_rot[3], v_rot[2]],
                 [v_rot[4], v_rot[5], v_rot[7], v_rot[6]],
                 [v_rot[0], v_rot[1], v_rot[5], v_rot[4]],
                 [v_rot[2], v_rot[3], v_rot[7], v_rot[6]],
                 [v_rot[0], v_rot[2], v_rot[6], v_rot[4]],
                 [v_rot[1], v_rot[3], v_rot[7], v_rot[5]]]
    poly.set_verts(faces_rot)
    poly.set_facecolor(['cyan', 'cyan', 'magenta', 'magenta', 'yellow', 'yellow'])
    
    # Update Arrows
    quiver_x.remove(); quiver_y.remove(); quiver_z.remove()
    bx = R_active @ np.array([1, 0, 0])
    by = R_active @ np.array([0, 1, 0])
    bz = R_active @ np.array([0, 0, 1])
    quiver_x = ax.quiver(0,0,0, bx[0], bx[1], bx[2], color='crimson', lw=3, arrow_length_ratio=0.2)
    quiver_y = ax.quiver(0,0,0, by[0], by[1], by[2], color='forestgreen', lw=3, arrow_length_ratio=0.2)
    quiver_z = ax.quiver(0,0,0, bz[0], bz[1], bz[2], color='royalblue', l