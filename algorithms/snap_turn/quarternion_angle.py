import numpy as np
import os

OUTPUT_DIR = "snap_turn/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def snapturn_quaternion(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    dot = np.dot(a, b)

    if np.isclose(dot, -1):
        # 180° case
        axis = np.cross(a, np.array([1, 0, 0]))
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, np.array([0, 1, 0]))
        axis = axis / np.linalg.norm(axis)
        return np.array([0, *axis])

    s = np.sqrt(2*(1+dot))
    xyz = np.cross(a,b)/s
    w = s/2
    return np.array([w,*xyz])

# ---------------------------------------------------------
# TEST
# ---------------------------------------------------------
a = np.array([1,0,0])
b = np.array([0,np.sqrt(3)/2,1/2])    # some random direction
b = b/np.linalg.norm(b)

q = snapturn_quaternion(a,b)
w = q[0]

theta_q = 2*np.arccos(w)
theta_geo = np.arccos(np.dot(a,b))

print("=== TEST 2: MINIMAL ANGLE ===")
print("Quaternion angle =", theta_q)
print("Geometric angle  =", theta_geo)

with open(os.path.join(OUTPUT_DIR,"test2_minimal_angle.txt"),"w") as f:
    f.write(f"Quaternion angle = {theta_q}\n")
    f.write(f"Geometric angle = {theta_geo}\n")

print("Saved test2_minimal_angle.txt")