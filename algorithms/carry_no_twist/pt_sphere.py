import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D
import os

OUTPUT_DIR = "MA_geometric-operators-on-manifolds/carry_no_twist/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------------------------------
# Sphere: X(u,v)
# -------------------------------------------------------
def X_sphere(u, v):
    return np.array([np.sin(u)*np.cos(v),
                     np.sin(u)*np.sin(v),
                     np.cos(u)])

def Xu_sphere(u, v):
    return np.array([np.cos(u)*np.cos(v),
                     np.cos(u)*np.sin(v),
                     -np.sin(u)])

def Xv_sphere(u, v):
    return np.array([-np.sin(u)*np.sin(v),
                      np.sin(u)*np.cos(v),
                      0.0])

# -------------------------------------------------------
# Christoffel symbols for sphere metric E=1, F=0, G=sin²u
# -------------------------------------------------------
def gamma_sphere(u, v):
    Gamma = np.zeros((2,2,2))
    Gamma[0,1,1] = -np.sin(u)*np.cos(u)     # Γ^u_{vv}
    Gamma[1,0,1] = np.cos(u)/np.sin(u)      # Γ^v_{uv}
    Gamma[1,1,0] = np.cos(u)/np.sin(u)      # Γ^v_{vu}
    return Gamma

# -------------------------------------------------------
# Curve: fixed latitude u=u0, v=t
# -------------------------------------------------------
u0 = 1.0                       # latitude (not too close to poles)
def u_of_t(t): return u0
def v_of_t(t): return t
def du_dt(t): return 0.0
def dv_dt(t): return 1.0

# -------------------------------------------------------
# Parallel transport ODE
# -------------------------------------------------------
def pt_ode_sphere(t, W):
    a, b = W
    u = u_of_t(t)
    v = v_of_t(t)
    du = du_dt(t)
    dv = dv_dt(t)
    Gamma = gamma_sphere(u, v)

    dW = np.zeros(2)
    for k in range(2):
        total = 0.0
        for i, Wi in enumerate([a, b]):
            for j, xj in enumerate([du, dv]):
                total += Gamma[k,i,j] * Wi * xj
        dW[k] = -total
    return dW

# -------------------------------------------------------
# Solve ODE
# -------------------------------------------------------
t0, t1 = 0, 2*np.pi
t_eval = np.linspace(t0, t1, 400)
W0 = np.array([0.0, 1.0])

sol = solve_ivp(pt_ode_sphere, (t0, t1), W0, t_eval=t_eval)
a, b = sol.y

# -------------------------------------------------------
# Build 3D geometry
# -------------------------------------------------------
U = np.full_like(t_eval, u0)
V = t_eval

pts = np.array([X_sphere(u, v) for u, v in zip(U, V)])
Xu_all = np.array([Xu_sphere(u, v) for u, v in zip(U, V)])
Xv_all = np.array([Xv_sphere(u, v) for u, v in zip(U, V)])
V3 = a[:,None] * Xu_all + b[:,None] * Xv_all

# Subsample arrows
idx = np.linspace(0, len(t_eval)-1, 18, dtype=int)

# -------------------------------------------------------
# Make pretty figure
# -------------------------------------------------------
fig = plt.figure(figsize=(12,6))
ax3d = fig.add_subplot(1,2,1,projection="3d")
ax2d = fig.add_subplot(1,2,2)

# Sphere surface
u_grid = np.linspace(0, np.pi, 60)
v_grid = np.linspace(0, 2*np.pi, 60)
U_grid, V_grid = np.meshgrid(u_grid, v_grid)
X_grid = np.sin(U_grid)*np.cos(V_grid)
Y_grid = np.sin(U_grid)*np.sin(V_grid)
Z_grid = np.cos(U_grid)

ax3d.plot_surface(X_grid, Y_grid, Z_grid,
                  rstride=2, cstride=2,
                  alpha=0.4, color="lightgray", edgecolor="none")

# Curve on sphere
ax3d.plot(pts[:,0], pts[:,1], pts[:,2], color="black", linewidth=2)

# Parallel transported vectors
for i in idx:
    P = pts[i]
    Vp = 0.25 * V3[i]
    ax3d.plot([P[0], P[0]+Vp[0]],
              [P[1], P[1]+Vp[1]],
              [P[2], P[2]+Vp[2]],
              color="blue", linewidth=2)

ax3d.set_title("Sphere: Parallel Transport Along Latitude")
ax3d.set_xticks([]); ax3d.set_yticks([]); ax3d.set_zticks([])
ax3d.view_init(elev=20, azim=130)

# ODE solution plot
ax2d.plot(t_eval, a, label="a(t)", linewidth=2)
ax2d.plot(t_eval, b, label="b(t)", linewidth=2)
ax2d.set_title("Solution of Parallel Transport ODE")
ax2d.set_xlabel("t")
ax2d.set_ylabel("components")
ax2d.grid(alpha=0.3)
ax2d.legend()

plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR,"pt_sphere.png")
plt.savefig(out_path, dpi=300)
plt.show()

print("Saved to:", out_path)