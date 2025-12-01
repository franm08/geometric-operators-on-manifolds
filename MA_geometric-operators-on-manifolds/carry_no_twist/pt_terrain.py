import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os

print("\n=== PARALLEL TRANSPORT ON SYNTHETIC TERRAIN ===")

OUTPUT_DIR = "MA_geometric-operators-on-manifolds/carry_no_twist/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------------------------
# Synthetic terrain function
# ----------------------------------------------
def h(u,v):  return 0.3*np.sin(u)*np.cos(v)
def h_u(u,v): return 0.3*np.cos(u)*np.cos(v)
def h_v(u,v): return -0.3*np.sin(u)*np.sin(v)

# Metric coefficients
def E(u,v): return 1 + h_u(u,v)**2
def F(u,v): return h_u(u,v)*h_v(u,v)
def G(u,v): return 1 + h_v(u,v)**2

def inv_metric(u,v):
    D = E(u,v)*G(u,v) - F(u,v)**2
    return G(u,v)/D, -F(u,v)/D, E(u,v)/D


# ----------------------------------------------
# Christoffel symbols
# ----------------------------------------------
def christoffel(u,v):
    # metric derivatives
    Eu = 2*h_u(u,v)*(-0.3*np.sin(u)*np.cos(v))
    Ev = 2*h_u(u,v)*(0.3*np.cos(u)*(-np.sin(v)))

    Fu = 0
    Fv = 0

    Gu = 2*h_v(u,v)*(0.3*np.cos(u)*np.cos(v))
    Gv = 2*h_v(u,v)*(-0.3*np.sin(u)*np.sin(v))

    g11, g12, g22 = inv_metric(u,v)

    Γ111 = 0.5*(g11*Eu + g12*(2*Fu - Ev))
    Γ112 = 0.5*(g11*Ev + g12*Gu)
    Γ122 = 0.5*(g11*(2*Fv - Gv) + g12*Gv)

    Γ211 = 0.5*(g12*Eu + g22*(2*Fu - Ev))
    Γ212 = 0.5*(g12*Ev + g22*Gu)
    Γ222 = 0.5*(g12*(2*Fv - Gv) + g22*Gv)

    return Γ111, Γ112, Γ122, Γ211, Γ212, Γ222



# ----------------------------------------------
# Define path in parameter space
# ----------------------------------------------
t = np.linspace(0, 1, 150)
u = -1 + 2*t
v = -1 + 2*t



# ----------------------------------------------
# Parallel transport ODE
# ----------------------------------------------
def ode(tt, W):
    i = int(tt*(len(t)-1))
    ui, vi = u[i], v[i]

    Γ111, Γ112, Γ122, Γ211, Γ212, Γ222 = christoffel(ui, vi)

    du = u[1]-u[0]
    dv = v[1]-v[0]

    a, b = W

    da = -(Γ111*a*du + 2*Γ112*a*dv + Γ122*b*dv)
    db = -(Γ211*a*du + 2*Γ212*a*dv + Γ222*b*dv)
    return [da, db]


print("\nODE System: W' = -Γ(u,v) · W")
print("Solving now...")

W0 = [1.0, 0.0]
sol = solve_ivp(ode, (0,1), W0, t_eval=t)

a = sol.y[0]
b = sol.y[1]

print("\nSolved a(t):", a[:10], "...")
print("Solved b(t):", b[:10], "...")



# ----------------------------------------------
# Compute transported vectors in 3D
# ----------------------------------------------
x = u
y = v
z = h(u,v)

Xu = np.column_stack([np.ones_like(u),
                       np.zeros_like(u),
                       h_u(u,v)])

Xv = np.column_stack([np.zeros_like(u),
                       np.ones_like(u),
                       h_v(u,v)])

Wx = a*Xu[:,0] + b*Xv[:,0]
Wy = a*Xu[:,1] + b*Xv[:,1]
Wz = a*Xu[:,2] + b*Xv[:,2]



# ----------------------------------------------
# Plot with printed values
# ----------------------------------------------
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')

# Terrain
U = np.linspace(-1,1,80)
V = np.linspace(-1,1,80)
UU, VV = np.meshgrid(U, V)
ZZ = h(UU, VV)

ax.plot_surface(UU, VV, ZZ, cmap='terrain', alpha=0.85)

# Path
ax.plot(x, y, z, 'k-', lw=2)

# Vectors
for i in range(0, len(x), 10):
    ax.quiver(x[i], y[i], z[i],
              0.5*Wx[i], 0.5*Wy[i], 0.5*Wz[i],
              color='blue')

# Annotate solution values
text = (
    f"a(0)={a[0]:.3f}, a(end)={a[-1]:.3f}\n"
    f"b(0)={b[0]:.5f}, b(end)={b[-1]:.5f}"
)
ax.text2D(0.05, 0.92, text, transform=ax.transAxes, fontsize=12)

ax.set_title("Parallel Transport on Synthetic Terrain")
ax.set_box_aspect([1,1,0.6])

# later…
out_path = os.path.join(OUTPUT_DIR, "pt_terrain.png")
plt.savefig(out_path, dpi=300)

plt.show()