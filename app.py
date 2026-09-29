import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
import plotly.express as px

# Page configuration
st.set_page_config(page_title="Matplotlib Generator", layout="centered")

st.title("Hose Calculator App by [ORUA](https://orua.auckland.ac.nz)")
st.write("Adjust the configuration on the left to update the hose solution and visualisation.")

# ---------------------------------------------------------
# 1. User Inputs (Sidebar Controls)
# ---------------------------------------------------------
st.sidebar.header("Hose Configuration")

val1 = st.sidebar.slider("Number of Rollers", min_value=8, max_value=25, value=10, step=1)
val2 = st.sidebar.slider("Disk Diameter (mm)", min_value=19, max_value=203, value=51, step=1)
val3 = st.sidebar.slider("Distance between Disks (mm)", min_value=45.0, max_value=65.0, value=65.0, step=1.0)
val4 = st.sidebar.slider("Target Pitch (mm)", min_value=0.0, max_value=20.0, value=10.5, step=0.1)

# ---------------------------------------------------------
# 2. Plot Generation Logic
# ---------------------------------------------------------
TOL = 1e-8
fig = plt.figure(figsize=(6, 6))

def closest_points_between_segments(p0, p1, q0, q1):
    """
    Finds the closest points between two line segments S1 (p0 to p1) and S2 (q0 to q1).
    Works in both 2D and 3D space.
    """
    u = p1 - p0
    v = q1 - q0
    w = p0 - q0
    
    a = np.dot(u, u)
    b = np.dot(u, v)
    c = np.dot(v, v)
    d = np.dot(u, w)
    e = np.dot(v, w)
    
    D = a * c - b * b
    sc, tc = 0.0, 0.0
    
    # Check if lines are parallel
    if D < 1e-8:
        sc = 0.0
        tc = d / b if b > c else e / c
    else:
        # Compute the parameters of the closest points on the infinite lines
        sc = (b * e - c * d) / D
        tc = (a * e - b * d) / D
        
    # Clamp sc to the segment S1 [0, 1]
    if sc < 0.0:
        sc = 0.0
        tc = e / c
    elif sc > 1.0:
        sc = 1.0
        tc = (b + e) / c
        
    # Clamp tc to the segment S2 [0, 1]
    if tc < 0.0:
        tc = 0.0
        sc = np.clip(-d / a, 0.0, 1.0)
    elif tc > 1.0:
        tc = 1.0
        sc = np.clip((b - d) / a, 0.0, 1.0)
        
    # Calculate the closest points using the clamped parameters
    closest_point_on_s1 = p0 + sc * u
    closest_point_on_s2 = q0 + tc * v
    
    return closest_point_on_s1, closest_point_on_s2

def simulate_hose(degrees_clockwise, num_rollers, diameter, dist_between_disks, target_pitch, draw=True):
    n = num_rollers
    d = diameter
    l = dist_between_disks
    alpha = degrees_clockwise / 360 * 2 * np.pi

    if draw:
        ax = fig.add_subplot(projection='3d')
    
    # Draw disk one
    center_x, center_y = 0.0, 0.0
    radius = d / 2.0

    theta = np.linspace(0, 2 * np.pi, 100)

    x = center_x + radius * np.cos(theta)
    y = center_y + radius * np.sin(theta)
    z = np.full(theta.shape, 0.0)  # Z stays constant

    if draw:
        ax.plot(x, y, z, color='blue', linewidth=2, label='Disk 1')

    # Draw disk two
    center_x, center_y = 0.0, 0.0
    radius = d / 2.0

    theta = np.linspace(0, 2 * np.pi, 100)

    xp = center_x + radius * np.cos(theta)
    yp = center_y + radius * np.sin(theta)
    zp = np.full(theta.shape, l)  # Z stays constant

    if draw:
        ax.plot(xp, yp, zp, color='cyan', linewidth=2, label='Disk 2')

    theta = np.linspace(0, 2 * np.pi, n)

    x = center_x + radius * np.cos(theta)
    y = center_y + radius * np.sin(theta)
    z = np.full(theta.shape, 0.0)  # Z stays constant

    xp = center_x + radius * np.cos(theta - alpha)
    yp = center_y + radius * np.sin(theta - alpha)
    zp = np.full(theta.shape, l)  # Z stays constant

    p = [None] * (n + 2)
    q = [None] * (n + 2)
    v = [None] * (n + 2)
    u = [None] * (n + 2)
    for i in range(0, n):
        if draw:
            ax.plot([x[i], xp[i]], [y[i], yp[i]], [z[i], zp[i]], color='red', linewidth=2)
        p[i] = np.array([x[i], y[i], z[i]])
        q[i] = np.array([xp[i], yp[i], zp[i]])
        v[i] = q[i] - p[i]
        u[i] = v[i] /  np.linalg.norm(v[i])
    p[n] = p[0]
    q[n] = q[0]
    v[n] = v[0]
    u[n] = u[0]
    p[n + 1] = p[1]
    q[n + 1] = q[1]
    v[n + 1] = v[1]
    u[n + 1] = u[1]

    A = [None] * (n + 1)
    b = [None] * (n + 1)
    s = [None] * (n + 2)
    s[0] = p[0]
    for i in range(0, n + 1):
        A[i] = np.zeros([4, 4])
        b[i] = np.zeros(4)
        A[i][0, 0] = u[i][0] # w_i, x normal
        A[i][0, 1] = u[i][1] # w_i, y normal
        A[i][0, 2] = u[i][2] # w_i, z normal
        b[i][0] = 0.0 # rhs normal
                        
        A[i][1, 0] = 1 # w_i, x intersection x
        A[i][1, 3] = - u[i + 1][0] # t_i intersection x
        b[i][1] = p[i + 1][0] - s[i][0] # rhs intersection x

        A[i][2, 1] = 1 # w_i, y intersection y
        A[i][2, 3] = - u[i + 1][1] # t_i intersection y
        b[i][2] = p[i + 1][1] - s[i][1] # rhs intersection y

        A[i][3, 2] = 1 # w_i, z intersection z
        A[i][3, 3] = - u[i + 1][2] # t_i intersection z
        b[i][3] = p[i + 1][2] - s[i][2] # rhs intersection z

        sol = np.linalg.solve(A[i], b[i])
        w = np.array([sol[0], sol[1], sol[2]])
        sw = s[i] + w
        t = sol[3]
        st = p[i + 1] + t * u[i + 1]

        dist = np.linalg.norm(sw - st)
#        print(i)
#        print(A[i])
#        print(b[i])
#        print(dist)
        assert(dist <= TOL)

        s[i + 1] = sw
        if draw:
            ax.plot([s[i][0], s[i + 1][0]], [s[i][1], s[i + 1][1]], [s[i][2], s[i + 1][2]],
                color='magenta', linewidth=2)


    r = p[0].copy()
    r[2] = q[0][2]
    if draw:
        ax.plot([p[0][0], r[0]], [p[0][1], r[1]], [p[0][2], r[2]], color='green', linewidth=2)
        ax.plot([p[0][0], r[0]], [p[0][1], r[1]], [p[0][2], target_pitch], color='black', linewidth=2)

    [st, pt] = closest_points_between_segments(s[n], s[n + 1], p[0], r)

    assert(abs(pt[0] - p[0][0]) <= TOL)
    assert(abs(pt[1] - p[0][1]) <= TOL)
    pitch_3d = np.linalg.norm(pt - p[0])
    pitch_2d = pt[2] - p[0][2]    
    assert(abs(pitch_3d - pitch_2d) <= TOL)
    simulated_pitch = pitch_3d

    diff = simulated_pitch - target_pitch
    print("DIFF = ", diff)

    if draw:
        # Label, etc
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')
        ax.set_title(f'Degrees clockwise = {degrees_clockwise:.4f}, Pitch error = {diff:.4f}')
        ax.legend()

        # Keep layout proportional so the circle doesn't look like an ellipse
        ax.set_box_aspect([1, 1, 1]) 

#        ax.view_init(elev=-45, azim=30, roll=75)
        ax.view_init(elev=-15, azim=20, roll=90)

        plt.show()

    return diff

def objective(variables, *fixed_values):
    degrees_clockwise = variables
    num_rollers, diameter, dist_between_disks, target_pitch = fixed_values

    diff = simulate_hose(degrees_clockwise, num_rollers, diameter, dist_between_disks, target_pitch, draw=False)

    return diff * diff

num_rollers        = val1
diameter           = val2
dist_between_disks = val3
target_pitch       = val4

constants = (num_rollers, diameter, dist_between_disks, target_pitch)

degrees_clockwise  = 10 # DIFF =  -0.7244619856623871
#degrees_clockwise  = 11 # DIFF =  0.216462153142011
#simulate_hose(degrees_clockwise, num_rollers, diameter, dist_between_disks, target_pitch)
result = minimize(
    fun  = objective,
    x0   = degrees_clockwise,
    args = constants
)
if result.success:
    optimized_x = result.x
    print(f"Optimization successful!")
    print(f"Optimized variables -> x: {optimized_x[0]:.4f}")
    print(f"Minimum function value: {result.fun:.4f}")
    degrees_clockwise = optimized_x[0]
    simulate_hose(degrees_clockwise, num_rollers, diameter, dist_between_disks, target_pitch)        

# Convert Matplotlib figure to HTML
st.plotly_chart(fig, use_container_width=True)