import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# -----------------------------
# Textbook Aesthetic Setup
# -----------------------------
# Set font to a serif style (like Times New Roman) for a formal look
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
# Reduced global font size
plt.rcParams['font.size'] = 10

# -----------------------------
# Helpers (Standard Rotation Matrices)
# -----------------------------
def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def get_circle_points(radius=1.0, n=360):
    t = np.linspace(0, 2*np.pi, n)
    return np.vstack([radius*np.cos(t), radius*np.sin(t), np.zeros_like(t)])

def plot_textbook_ring(ax, R_matrix, radius, color, label, lw=3):
    P_local = get_circle_points(radius=radius)
    P_world = R_matrix @ P_local
    
    # Draw ring with thinner lines for a printed look
    ax.plot(P_world[0], P_world[1], P_world[2], 
            color=color, linewidth=lw, alpha=0.9)

    # Place label. Index 45 puts it at a 45-degree angle.
    idx = 45
    # Increase offset slightly for clear separation
    lx, ly, lz = P_world[:, idx] * 1.15
    
    # Text is now BLACK and uses serif font.
    # REDUCED FONT SIZE here.
    ax.text(lx, ly, lz, label, color='black', fontsize=11, 
            ha='center', va='center', zorder=10)

# -----------------------------
# Configuration: Gimbal Lock
# -----------------------------
yaw_angle   = 0.0
pitch_angle = np.pi / 2 # Pitch is exactly 90 degrees
roll_angle  = 0.0

# 1. Outer Ring (Yaw - Z axis)
R_outer = rot_z(yaw_angle)

# 2. Middle Ring (Pitch - Y axis relative to outer)
R_mid_base = rot_x(np.pi/2) 
R_mid = R_outer @ R_mid_base

# 3. Inner Ring (Roll - X axis relative to middle)
R_inner_base = rot_y(np.pi/2)
R_inner = R_outer @ rot_y(pitch_angle) @ R_inner_base

# -----------------------------
# Visualization
# -----------------------------
# REDUCED FIGURE SIZE here.
fig = plt.figure(figsize=(4, 5), dpi=300, facecolor='white')
ax = fig.add_subplot(111, projection='3d')
ax.set_axis_off()
ax.set_box_aspect([1, 1, 1]) 

# Aesthetics - Slightly darker, more formal colors
c_outer = '#004488' # Dark Blue
c_mid   = '#008844' # Dark Green
c_inner = '#BB2222' # Dark Red
r1, r2, r3 = 1.3, 1.0, 0.7
line_width = 2.5 # Slightly thinner lines for smaller figure

# Plot Rings with black, descriptive serif labels
plot_textbook_ring(ax, R_outer, r1, c_outer, r"$\gamma$", lw=line_width)
plot_textbook_ring(ax, R_mid,   r2, c_mid,   r"$\beta=90^{\circ}$", lw=line_width)
plot_textbook_ring(ax, R_inner, r3, c_inner, r"$\alpha$", lw=line_width)

# Add a dashed line to emphasize the aligned axes
ax.plot([0, 0], [0, 0], [-1.6, 1.6], 'k--', linewidth=1.5, alpha=0.6, zorder=1)
# REMOVED THE "Aligned Axes" TEXT HERE.

# Central pivot
ax.scatter([0], [0], [0], color='black', s=30, alpha=0.6) # Smaller dot

# Set View
ax.view_init(elev=22, azim=45)
lim = 0.8
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_zlim(-lim, lim)

# Save with a white background
plt.savefig("gimbal_lock_static.png", bbox_inches='tight', facecolor=fig.get_facecolor())
plt.show()