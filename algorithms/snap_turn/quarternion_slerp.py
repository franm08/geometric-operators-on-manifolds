import numpy as np
import os

OUTPUT_DIR = "snap_turn/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def snapturn_quaternion(a,b):
    a=a/np.linalg.norm(a)
    b=b/np.linalg.norm(b)
    dot=np.dot(a,b)

    if np.isclose(dot,-1):
        axis=np.cross(a,[1,0,0])
        if np.linalg.norm(axis)<1e-6:
            axis=np.cross(a,[0,1,0])
        axis=axis/np.linalg.norm(axis)
        return np.array([0,*axis])

    s=np.sqrt(2*(1+dot))
    xyz=np.cross(a,b)/s
    w=s/2
    return np.array([w,*xyz])

def qmul(q1,q2):
    w1,x1,y1,z1=q1
    w2,x2,y2,z2=q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def qconj(q): return np.array([q[0],-q[1],-q[2],-q[3]])

def qrot(q, v):
    return qmul(q, qmul(np.hstack(([0],v)), qconj(q)))[1:]

# ---------------------------------------------------------
# TEST
# ---------------------------------------------------------
a = np.array([1,0,0])
b = np.array([0,1,0])
q = snapturn_quaternion(a,b)

# Generate SLERP points
num = 20
t_vals = np.linspace(0,1,num)
points = []

for t in t_vals:
    # SLERP formula
    w = q[0]
    theta = np.arccos(w)
    axis = q[1:]/np.sqrt(1-w*w)

    # rotation by angle t*theta
    qt = np.array([np.cos(t*theta),
                   *(axis*np.sin(t*theta))])
    points.append(qrot(qt,a))

points = np.array(points)

print("=== TEST 4: SLERP GREAT-CIRCLE ===")
print("All SLERP points lie on S^2 (unit sphere)?", 
      np.allclose(np.linalg.norm(points,axis=1),1,atol=1e-6))

np.savetxt(os.path.join(OUTPUT_DIR,"test4_slerp_points.txt"),points)
print("Saved test4_slerp_points.txt")