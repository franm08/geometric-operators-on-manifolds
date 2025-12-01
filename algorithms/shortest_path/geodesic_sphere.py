import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
import os

os.makedirs("outputs", exist_ok=True)

# --------------------------------------------------------------
#   GEODESIC ON SPHERE
# --------------------------------------------------------------

def geodesic_sphere_ode(t, y):
    u, v, du, dv = y
    if u <= 1e-3 or u >= np.pi-1e-3:
        return [0,0,0,0]

    Gamma_u_vv = -np.sin(u)*np.cos(u)
    Gamma_v_uv = 1/np.tan(u)

    d2u = -Gamma_u_vv * dv*dv
    d2v = -2*Gamma_v_uv * du * dv
    return [du,dv,d2u,d2v]

u0, v0 = np.pi/2, 0.0
du0, dv0 = 0.0, 1.0
L = np.sqrt(du0**2 + dv0**2)
du0/=L; dv0/=L
y0 = [u0,v0,du0,dv0]

sol = solve_ivp(geodesic_sphere_ode, (0,6), y0, t_eval=np.linspace(0,6,800))
u, v = sol.y[0], sol.y[1]

x = np.sin(u)*np.cos(v)
y = np.sin(u)*np.sin(v)
z = np.cos(u)

A = np.array([x[0],y[0],z[0]])
B = np.array([x[-1],y[-1],z[-1]])
T = np.array([x[1]-x[0], y[1]-y[0], z[1]-z[0]])
T = T/np.linalg.norm(T)*0.25

phi = np.linspace(0,np.pi,80)
theta = np.linspace(0,2*np.pi,80)
phi,theta = np.meshgrid(phi,theta)
Xs = np.sin(phi)*np.cos(theta)
Ys = np.sin(phi)*np.sin(theta)
Zs = np.cos(phi)

fig = plt.figure(figsize=(7,6))
ax = fig.add_subplot(111,projection="3d")

ax.plot_surface(Xs,Ys,Zs,alpha=0.25,color="lightblue",edgecolor="none")
ax.plot(x,y,z,'r',linewidth=3)

ax.scatter(A[0],A[1],A[2],color='blue',s=60)
ax.text(A[0],A[1],A[2],"  Point A",color='blue')

ax.scatter(B[0],B[1],B[2],color='green',s=60)
ax.text(B[0],B[1],B[2],"  Point B",color='green')

ax.set_xlim(-1.1,1.1); ax.set_ylim(-1.1,1.1); ax.set_zlim(-1.1,1.1)
ax.view_init(25,235)
plt.tight_layout()
plt.savefig("shortest_path/outputs/geodesic_sphere.png", dpi=300)
plt.show()

# Animation
do_animation=True
if do_animation:
    fig2=plt.figure(figsize=(7,6))
    ax2=fig2.add_subplot(111,projection="3d")

    ax2.plot_surface(Xs,Ys,Zs,alpha=0.25,color="lightblue",edgecolor="none")
    geod_line,=ax2.plot([],[],[],'r',linewidth=3)

    ax2.scatter(A[0],A[1],A[2],color='blue',s=60)
    ax2.quiver(A[0],A[1],A[2],T[0],T[1],T[2],
               color='black',linewidth=2,arrow_length_ratio=0.2)

    ax2.set_xlim(-1.1,1.1); ax2.set_ylim(-1.1,1.1); ax2.set_zlim(-1.1,1.1)
    ax2.view_init(25,235)

    def update(frame):
        geod_line.set_data(x[:frame],y[:frame])
        geod_line.set_3d_properties(z[:frame])
        return geod_line,

    ani=FuncAnimation(fig2,update,frames=len(x),interval=15,blit=True)
    ani.save("outputs/geodesic_sphere.gif",writer="pillow",fps=60)