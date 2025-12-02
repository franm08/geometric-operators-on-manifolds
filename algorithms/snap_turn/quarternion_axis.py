import numpy as np
import os

OUTPUT_DIR = "snap_turn/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def snapturn_quaternion(a,b):
    a = a/np.linalg.norm(a)
    b = b/np.linalg.norm(b)
    dot = np.dot(a,b)

    if np.isclose(dot,-1):
        axis = np.cross(a,[1,0,0])
        if np.linalg.norm(axis)<1e-6:
            axis = np.cross(a,[0,1,0])
        axis = axis/np.linalg.norm(axis)
        return np.array([0,*axis])

    s = np.sqrt(2*(1+dot))
    xyz = np.cross(a,b)/s
    w = s/2
    return np.array([w,*xyz])

# ---------------------------------------------------------
# TEST
# ---------------------------------------------------------
a = np.array([1,0,0])
b = np.array([0,0,1])
a /= np.linalg.norm(a)
b /= np.linalg.norm(b)

q = snapturn_quaternion(a,b)

axis_q = q[1:]
axis_q = axis_q/np.linalg.norm(axis_q)

axis_geo = np.cross(a,b)
axis_geo = axis_geo/np.linalg.norm(axis_geo)

print("=== TEST 3: AXIS AGREEMENT ===")
print("Quaternion axis =", axis_q)
print("Geometric axis  =", axis_geo)

with open(os.path.join(OUTPUT_DIR,"test3_axis.txt"),"w") as f:
    f.write(f"Quaternion axis = {axis_q}\n")
    f.write(f"Geometric axis = {axis_geo}\n")

print("Saved test3_axis.txt")