# pt_plane.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os

# -------------------------------------------------------
# Setup
# -------------------------------------------------------
OUTPUT_DIR = "MA_geometric-operators-on-manifolds/carry_no_twist/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------------------------------
# Plane surface
# -------------------------------------------------------
def X_plane(u, v):
    return np.array([u, v, 0.0])

def Xu_plane(u, v):
    return np.array([1.0, 0.0, 0.0])

def Xv_plane(u, v):
    return np.array([0.0, 1.0, 0.0])

# -------------------------------------------------------
# Curve: straight line from -2 to 2 in u, v = 0
# -------------------------------------------------------
t0, t1 = 0.0, 1.0

def u_of_t(t):  return -2.0 + 4.0 * t
def v_of_t(t):  return 0.0
def du_dt(t):   return 4.0
def dv_dt(t):   return 0.0

# Christoffel = 0 on plane
def gamma_plane(u, v):
    return np.zeros((2, 2, 2))

# -------------------------------------------------------
# Parallel Transport ODE: W' = −Γ * W * x'
# W = (a,b)
# -------------------------------------------------------
def pt_ode_plane(t, W):
    # a,b components
    a, b = W
    u = u_of_t(t)
    v = v_of_t(t)
    du = du_dt(t)
    dv = dv_dt(t)

    Gamma = gamma_plane(u, v)  # all zeros

    dW = np.zeros(2)
    for k in range(2):
        s = 0.0
        for i in range(2):
            Wi = a if i == 0 else b
            for j in range(2):
                xdot = du if j == 0 else dv
                s += Gamma[k,i,j] * Wi * xdot
        dW[k] = -s

    return dW

# Initial vector W(0)
W0 = np.array([1.0, 0.0])

# Solve
t_eval = np.linspace(t0, t1, 200)
sol = solve_ivp(pt_ode_plane, (t0, t1), W0, t_eval=t_eval)

a = sol.y[0]
b = sol.y[1]

print("=== PARALLEL TRANSPORT ON PLANE ===")
print("First 10 a(t):", a[:10])
print("First 10 b(t):", b[:10])
print("Since Γ = 0, ODE → a'=0, b'=0. Vector stays constant.")

# -------------------------------------------------------
# Build Geometry (correct!)
# -------------------------------------------------------
U = np.array([u_of_t(t) for t in t_eval])
V = np.array([v_of_t(t) for t in t_eval])

pts = np.array([X_plane(u, v) for u, v in zip(U, V)])

Xu_all = np.array([Xu_plane(u,v) for u,v in zip(U,V)])
Xv_all = np.array([Xv_plane(u,v) for u,v in zip(U,V)])

# Construct world tangent vectors
V3 = a[:,None] * Xu_all + b[:,None] * Xv_all

# Choose arrow locations
idx = np.linspace(0, len(t_eval)-1, 12, dtype=int)

# -------------------------------------------------------
# Plot
# -------------------------------------------------------
fig = plt.figure(figsize=(9,4))
ax3d = fig.add_subplot(1,2,1, projection="3d")
ax2d = fig.add_subplot(1,2,2)

# surface patch
u_grid = np.linspace(-2.5, 2.5, 15)
v_grid = np.linspace(-1.0, 1.0, 15)
U_grid, V_grid = np.meshgrid(u_grid, v_grid)
Z_grid = np.zeros_like(U_grid)

ax3d.plot_surface(U_grid, V_grid, Z_grid,
                  color="lightgray", alpha=0.6, edgecolor="none")

# geodesic path
ax3d.plot(pts[:,0], pts[:,1], pts[:,2], color="black", linewidth=3)

# arrows
for i in idx:
    P = pts[i]
    Vp = V3[i]
    ax3d.quiver(P[0], P[1], P[2],
                Vp[0], Vp[1], Vp[2],
                length=0.4, color="blue")

# zoom tight
ax3d.set_xlim(min(U)-0.2, max(U)+0.2)
ax3d.set_ylim(min(V)-0.2, max(V)+0.2)
ax3d.set_zlim(-0.3, 0.3)

# clean style
ax3d.set_xticks([]); ax3d.set_yticks([]); ax3d.set_zticks([])
ax3d.set_title("Plane: Parallel Transport (vectors constant)")

# 2D ODE solutions
ax2d.plot(t_eval, a, label="a(t)", linewidth=2)
ax2d.plot(t_eval, b, label="b(t)", linewidth=2)
ax2d.set_title("Solution to Parallel Transport ODE")
ax2d.set_xlabel("t"); ax2d.set_ylabel("components")
ax2d.grid(alpha=0.3)
ax2d.legend()

plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, "pt_plane.png")
plt.savefig(out_path, dpi=300)
print("Saved to:", out_path)

plt.show()