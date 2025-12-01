import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
import os

os.makedirs("outputs", exist_ok=True)

# --------------------------------------------------------------
#   GEODESIC ON THE PLANE: X(u,v) = (u, v, 0)
# --------------------------------------------------------------

def geodesic_plane_ode(t, y):
    u, v, du, dv = y
    if abs(u) > 3 or abs(v) > 3:
        return [0,0,0,0]
    return [du, dv, 0.0, 0.0]

# Initial condition (normalize tangent)
u0, v0 = -1.0, -1.0
du0, dv0 = 1.0, 0.6
L = np.sqrt(du0**2 + dv0**2)
du0 /= L; dv0 /= L
y0 = [u0, v0, du0, dv0]

# Integrate
t_eval = np.linspace(0,5,600)
sol = solve_ivp(geodesic_plane_ode, (0,5), y0, t_eval=t_eval)
u, v = sol.y[0], sol.y[1]
x, y, z = u, v, np.zeros_like(u)

# Points A and B
A = np.array([x[0],  y[0],  z[0]])
B = np.array([x[-1], y[-1], z[-1]])

# Tangent vector at A
T = np.array([x[1]-x[0], y[1]-y[0], z[1]-z[0]])
T = T / np.linalg.norm(T) * 0.4

# Plane mesh
U = np.linspace(-3,3,40)
V = np.linspace(-3,3,40)
U, V = np.meshgrid(U, V)
Z = np.zeros_like(U)

# Static plot
fig = plt.figure(figsize=(7,6))
ax = fig.add_subplot(111, projection="3d")

ax.plot_surface(U, V, Z, alpha=0.25, color="lightgray", edgecolor="none")
ax.plot(x, y, z, 'r', linewidth=3)

# Point A / B and tangent vector
ax.scatter(A[0],A[1],A[2],color='blue',s=60)
ax.text(A[0],A[1],A[2],"  Point A",color='blue')
ax.scatter(B[0],B[1],B[2],color='green',s=60)
ax.text(B[0],B[1],B[2],"  Point B",color='green')

ax.set_xlim(-3,3); ax.set_ylim(-3,3); ax.set_zlim(-0.5,0.5)
ax.view_init(35,225)
plt.tight_layout()
plt.savefig("shortest_path/outputs/geodesic_plane.png", dpi=300)
plt.show()

# Animation
do_animation = True
if do_animation:
    fig2 = plt.figure(figsize=(7,6))
    ax2 = fig2.add_subplot(111, projection="3d")

    ax2.plot_surface(U, V, Z, alpha=0.25, color="lightgray", edgecolor="none")
    geod_line, = ax2.plot([],[],[],'r',linewidth=3)

    ax2.scatter(A[0],A[1],A[2],color='blue',s=60)
    ax2.quiver(A[0],A[1],A[2], T[0],T[1],T[2],
               color='black', linewidth=2, arrow_length_ratio=0.2)

    ax2.set_xlim(-3,3); ax2.set_ylim(-3,3); ax2.set_zlim(-0.5,0.5)
    ax2.view_init(35,225)

    def update(frame):
        geod_line.set_data(x[:frame], y[:frame])
        geod_line.set_3d_properties(z[:frame])
        return geod_line,

    ani = FuncAnimation(fig2, update, frames=len(x), interval=15, blit=True)
    ani.save("outputs/geodesic_plane.gif", writer="pillow", fps=60)