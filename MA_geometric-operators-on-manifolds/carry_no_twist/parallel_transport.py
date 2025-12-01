import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp

# ============================================================
# 1. Sphere parametrization
# ============================================================

def X(u, v):
    return np.array([
        np.sin(u)*np.cos(v),
        np.sin(u)*np.sin(v),
        np.cos(u)
    ])

def Xu(u, v):
    return np.array([
        np.cos(u)*np.cos(v),
        np.cos(u)*np.sin(v),
        -np.sin(u)
    ])

def Xv(u, v):
    return np.array([
        -np.sin(u)*np.sin(v),
        np.sin(u)*np.cos(v),
        0
    ])

# ============================================================
# 2. Christoffel symbols (sphere)
# ============================================================

def christoffel(u, v):
    Γ111 = 0
    Γ112 = 0
    Γ122 = -np.sin(u)*np.cos(u)

    Γ211 = 0
    Γ212 = np.cos(u)/np.sin(u)
    Γ222 = 0

    return Γ111, Γ112, Γ122, Γ211, Γ212, Γ222

# ============================================================
# 3. Parallel transport ODE
# ============================================================

def ode(t, Y):
    u, v, du, dv, a, b = Y
    Γ111, Γ112, Γ122, Γ211, Γ212, Γ222 = christoffel(u, v)

    da = -(a*Γ111*du + a*Γ112*dv + b*Γ211*du + b*Γ212*dv)
    db = -(a*Γ112*du + a*Γ122*dv + b*Γ212*du + b*Γ222*dv)

    return [du, dv, 0, 0, da, db]

# ============================================================
# 4. Choose a meridian path
# ============================================================

v = 0
u_vals = np.linspace(0.2, 2.6, 200)    # nice clean arc
du_vals = np.gradient(u_vals)
dv_vals = 0*u_vals                     # constant longitude

# ============================================================
# 5. Solve parallel transport
# ============================================================

a0, b0 = 1, 0

sol = solve_ivp(
    ode,
    (0, 1),
    [u_vals[0], v, du_vals[0], 0, a0, b0],
    t_eval=np.linspace(0, 1, len(u_vals))
)

u = sol.y[0]
v = sol.y[1]
a = sol.y[4]
b = sol.y[5]

# ============================================================
# 6. Build vectors
# ============================================================

points = np.array([X(uu, vv) for uu, vv in zip(u, v)])
W_par = np.array([
    a[i]*Xu(u[i], v[i]) + b[i]*Xv(u[i], v[i])
    for i in range(len(u))
])
W_par /= np.linalg.norm(W_par, axis=1)[:,None]

# WRONG Euclidean: keep initial vector and project to tangent
wrong0 = (a0*Xu(u[0], v[0]) + b0*Xv(u[0], v[0]))
W_wrong = []
for i in range(len(points)):
    n = points[i]
    proj = wrong0 - np.dot(wrong0, n)*n
    W_wrong.append(proj/np.linalg.norm(proj))

W_wrong = np.array(W_wrong)

# ============================================================
# 7. Plot
# ============================================================

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')

# Sphere
phi = np.linspace(0, np.pi, 60)
theta = np.linspace(0, 2*np.pi, 60)
PHI, THETA = np.meshgrid(phi, theta)
XS = np.sin(PHI)*np.cos(THETA)
YS = np.sin(PHI)*np.sin(THETA)
ZS = np.cos(PHI)

ax.plot_surface(XS, YS, ZS, alpha=0.3, linewidth=0)

# Parallel transport (black)
for i in range(0, len(points), 15):
    ax.quiver(*points[i], *(0.3*W_par[i]), color='black')

# Wrong Euclidean derivative (red)
for i in range(0, len(points), 15):
    ax.quiver(*points[i], *(0.3*W_wrong[i]), color='red')

ax.set_axis_off()
plt.show()