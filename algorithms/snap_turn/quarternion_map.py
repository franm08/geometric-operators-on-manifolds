import numpy as np
import os

# ---------------------------------------------------------
# Setup output directory
# ---------------------------------------------------------
OUTPUT_DIR = "snap_turn/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------
# Minimal rotation quaternion a→b
# ---------------------------------------------------------
def snapturn_quaternion(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)

    dot = np.dot(a, b)

    if np.isclose(dot, -1.0):
        # 180° case
        axis = np.array([1, 0, 0])
        if np.allclose(a, axis):
            axis = np.array([0, 1, 0])
        axis = axis - np.dot(axis, a) * a
        axis /= np.linalg.norm(axis)
        return np.array([0.0, *axis])

    s = np.sqrt(2.0 * (1.0 + dot))
    xyz = np.cross(a, b) / s
    w = s / 2.0
    return np.array([w, xyz[0], xyz[1], xyz[2]])

# Quaternion rotation q * x * q^{-1}
def qrot(q, x):
    w, xq, yq, zq = q
    q_vec = np.array([xq, yq, zq])
    t = 2 * np.cross(q_vec, x)
    return x + w * t + np.cross(q_vec, t)

# ---------------------------------------------------------
# TEST CASE
# ---------------------------------------------------------
a = np.array([1.0, 0.0, 0.0])
b = np.array([0.0, 1.0, 0.0])
q = snapturn_quaternion(a, b)
R_a = qrot(q, a)

print("=== TEST 1: CORRECTNESS ===")
print("a =", a)
print("b =", b)
print("q =", q)
print("R_q(a) =", R_a)

outpath = os.path.join(OUTPUT_DIR, "test1_correctness.txt")
with open(outpath, "w") as f:
    f.write(f"a = {a}\n")
    f.write(f"b = {b}\n")
    f.write(f"q = {q}\n")
    f.write(f"R_q(a) = {R_a}\n")

print("Saved:", outpath)