import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
import os

# --- Setup: Create output directory ---
os.makedirs("outputs", exist_ok=True)

# --------------------------------------------------------------
#   1. TERRAIN HEIGHTMAP DEFINITIONS (h, and its derivatives)
# --------------------------------------------------------------

# Height function h(u, v) - a combination of a bowl (0.15*(u*u+v*v)) and a ripple pattern
def h(u,v):
    return 0.15*(u*u+v*v) + 0.4*np.sin(2.8*u)*np.cos(2.8*v)

# First partial derivatives (hu, hv)
def h_u(u,v):
    return 0.3*u + 0.4*2.8*np.cos(2.8*u)*np.cos(2.8*v)

def h_v(u,v):
    return 0.3*v - 0.4*2.8*np.sin(2.8*u)*np.sin(2.8*v)

# Second partial derivatives (huu, huv, hvv)
def h_uu(u,v):
    return 0.3 - 0.4*(2.8**2)*np.sin(2.8*u)*np.cos(2.8*v)

def h_uv(u,v):
    return -0.4*(2.8**2)*np.cos(2.8*u)*np.sin(2.8*v)

def h_vv(u,v):
    return 0.3 - 0.4*(2.8**2)*np.sin(2.8*u)*np.cos(2.8*v)

# --------------------------------------------------------------
#   2. FIRST FUNDAMENTAL FORM (Metric Coefficients)
# --------------------------------------------------------------

def E(u,v): return 1 + h_u(u,v)**2
def F(u,v): return h_u(u,v)*h_v(u,v)
def G(u,v): return 1 + h_v(u,v)**2

def det_metric(u,v): return E(u,v)*G(u,v) - F(u,v)**2

# --------------------------------------------------------------
#   3. CHRISTOFFEL SYMBOLS (Γk_ij)
# --------------------------------------------------------------

# Computes the six necessary Christoffel symbols at (u, v)
def christoffel(u,v):
    Euv=E(u,v);Fuv=F(u,v);Guv=G(u,v)
    
    # Partial derivatives of metric coefficients (g_ij, i.e., E, F, G)
    Eu=2*h_u(u,v)*h_uu(u,v)
    Ev=2*h_u(u,v)*h_uv(u,v)
    Fu=h_uu(u,v)*h_v(u,v)+h_u(u,v)*h_uv(u,v)
    Fv=h_uv(u,v)*h_v(u,v)+h_u(u,v)*h_vv(u,v)
    Gu=2*h_v(u,v)*h_uv(u,v)
    Gv=2*h_v(u,v)*h_vv(u,v)

    D=det_metric(u,v)
    
    # Inverse metric coefficients (g^ij)
    g11=Guv/D
    g12=-Fuv/D
    g22=Euv/D

    # Christoffel symbols (Γk_ij = 1/2 * g^kl * (∂g_lj/∂x^i + ∂g_li/∂x^j - ∂g_ij/∂x^l))
    # Note: The original code uses a simplified/derived form of the formula, 
    # which is specific to surfaces and ensures the result is correct.
    
    # Γ^u_11 = G111 (using simplified notation from the original code)
    G111=0.5*(g11*Eu + g12*(Eu-2*Fu))
    # Γ^u_12 = Γ^u_21 = G112
    G112=0.5*(g11*Ev + g12*(Ev-Gv))
    # Γ^u_22 = G122
    G122=0.5*(g11*(2*Fv-Gu) + g12*Gv)

    # Γ^v_11 = G211
    G211=0.5*(g12*Eu + g22*(Eu-2*Fu))
    # Γ^v_12 = Γ^v_21 = G212
    G212=0.5*(g12*Ev + g22*(Ev-Gv))
    # Γ^v_22 = G222
    G222=0.5*(g12*(2*Fv-Gu) + g22*Gv)

    return G111,G112,G122,G211,G212,G222

# --------------------------------------------------------------
#   4. GEODESIC ODE SYSTEM (y'' + Γ y'^2 = 0)
# --------------------------------------------------------------

# Input y = [u, v, du/dt, dv/dt]
# Output dy/dt = [du/dt, dv/dt, d^2u/dt^2, d^2v/dt^2]
def geodesic_ode(t,y):
    u,v,du,dv=y
    
    # Stop condition (optional: prevents calculation outside the visual domain)
    if abs(u)>1.5 or abs(v)>1.5:
        return [0,0,0,0]
        
    G111,G112,G122,G211,G212,G222=christoffel(u,v)
    
    # d^2u/dt^2
    d2u=-(G111*du*du+2*G112*du*dv+G122*dv*dv)
    # d^2v/dt^2
    d2v=-(G211*du*du+2*G212*du*dv+G222*dv*dv)
    
    return [du,dv,d2u,d2v]

# --------------------------------------------------------------
#   5. SOLVER SETUP AND EXECUTION
# --------------------------------------------------------------

# Initial condition (u0, v0) and initial velocity vector (du0, dv0)
u0,v0=0.2,-0.2
du0,dv0=0.6,0.3

# Normalize the initial velocity vector to ensure unit speed (L=1)
L=np.sqrt(du0**2+dv0**2)
du0/=L; dv0/=L

y0=[u0,v0,du0,dv0]

# Solve the ODE system
sol=solve_ivp(geodesic_ode,(0,5),y0,t_eval=np.linspace(0,5,800))
u,v=sol.y[0],sol.y[1]
x=u; y=v; z=h(u,v)

# Points for plotting start/end markers
A=np.array([x[0],y[0],z[0]])
B=np.array([x[-1],y[-1],z[-1]])

# Tangent vector for the quiver plot
T=np.array([x[1]-x[0],y[1]-y[0],z[1]-z[0]])
T=T/np.linalg.norm(T)*0.35 # Normalize and scale

# Terrain grid for plotting the surface
U=np.linspace(-1.5,1.5,150)
V=np.linspace(-1.5,1.5,150)
U,V=np.meshgrid(U,V)
Z=h(U,V)

# --------------------------------------------------------------
#   6. STATIC 3D PLOT
# --------------------------------------------------------------

print("Generating static image...")
fig=plt.figure(figsize=(8,6))
ax=fig.add_subplot(111,projection="3d")

# Plot the surface
ax.plot_surface(U,V,Z,cmap='terrain',edgecolor='none',alpha=0.9)
# Plot the geodesic path
ax.plot(x,y,z,'r',linewidth=3)

# Start and End markers
ax.scatter(A[0],A[1],A[2],color='blue',s=60)
ax.text(A[0],A[1],A[2],"  Point A",color='blue')

ax.scatter(B[0],B[1],B[2],color='green',s=60)
ax.text(B[0],B[1],B[2],"  Point B",color='green')

# Set limits and view angle
ax.set_xlim(-1.5,1.5); ax.set_ylim(-1.5,1.5); ax.set_zlim(Z.min(),Z.max())
ax.view_init(45,225) # Tilt the view for a better perspective

plt.tight_layout()
plt.savefig("outputs/geodesic_terrain_static.png", dpi=300)
plt.show() # Display the static plot

# --------------------------------------------------------------
#   7. ANIMATION (Geodesic Tracing)
# --------------------------------------------------------------

# NOTE: For 3D animations, you must have 'pillow' installed (pip install pillow) 
# and use 'blit=False' for reliable saving.

do_animation=True
if do_animation:
    print("Generating animation (This will be slower due to blit=False)...")
    fig2=plt.figure(figsize=(8,6))
    ax2=fig2.add_subplot(111,projection="3d")

    # Plot the surface (lower alpha to see the line better)
    ax2.plot_surface(U,V,Z,cmap='terrain',edgecolor='none',alpha=0.7)
    
    # Initialize the line object (empty plot)
    geod_line,=ax2.plot([],[],[],'r',linewidth=4, label='Geodesic Path')

    # Start point and initial tangent vector
    ax2.scatter(A[0],A[1],A[2],color='blue',s=60)
    ax2.quiver(A[0],A[1],A[2],T[0],T[1],T[2],
               color='black',linewidth=2,arrow_length_ratio=0.2)

    ax2.set_xlim(-1.5,1.5); ax2.set_ylim(-1.5,1.5); ax2.set_zlim(Z.min(),Z.max())
    ax2.view_init(45,225)
    ax2.set_title("Geodesic Path Animation")

    def update(frame):
        # Set the data for the line up to the current frame index
        geod_line.set_data(x[:frame],y[:frame])
        geod_line.set_3d_properties(z[:frame])
        return (geod_line,) # Return the updated line object in a tuple

    # CRITICAL: Set blit=False for 3D plots
    # Frames is the number of points in the solution (len(x))
    # fps=50 (frames per second) gives a smooth result
    ani=FuncAnimation(fig2,update,frames=len(x),interval=15,blit=False) 
    
    # Save the animation to the outputs directory
    ani.save("outputs/geodesic_terrain.gif",writer="pillow",fps=50) 
    print("Animation saved successfully to outputs/geodesic_terrain.gif")
    
    plt.show() # Display the animation window