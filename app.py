import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
import plotly.graph_objects as go

TOWARDS_TURN = True

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
fig = go.Figure()

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
    alpha = degrees_clockwise / 360.0 * 2 * np.pi
   
    # Draw disk one
    center_x, center_y = 0.0, 0.0
    radius = d / 2.0

    theta = np.linspace(0, 2 * np.pi, 100)

    x = center_x + radius * np.cos(theta)
    y = center_y + radius * np.sin(theta)
    z = np.full(theta.shape, 0.0)  # Z stays constant

    if draw:
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='lines',
            line=dict(color='blue', width=4),
            name='Disk 1'
        ))

    # Draw disk two
    center_x, center_y = 0.0, 0.0
    radius = d / 2.0

    theta = np.linspace(0, 2 * np.pi, 100)

    xp = center_x + radius * np.cos(theta)
    yp = center_y + radius * np.sin(theta)
    zp = np.full(theta.shape, l)  # Z stays constant

    if draw:
        fig.add_trace(go.Scatter3d(
            x=xp, y=yp, z=zp,
            mode='lines',
            line=dict(color='cyan', width=4),
            name='Disk 2'
        ))

    theta = np.linspace(0, 2 * np.pi, n)

    x = center_x + radius * np.cos(theta)
    y = center_y + radius * np.sin(theta)
    z = np.full(theta.shape, 0.0)  # Z stays constant

    if TOWARDS_TURN:
      # Winding towards the turn
      z = np.full(theta.shape, l)  # Z stays constant
      xp = center_x + radius * np.cos(theta + alpha)
      yp = center_y + radius * np.sin(theta + alpha)
    else:
      # Winding away from the turn
      xp = center_x + radius * np.cos(theta - alpha)
      yp = center_y + radius * np.sin(theta - alpha)
      zp = np.full(theta.shape, l)  # Z stays constant


    eta = np.linspace(0, 4 * np.pi, 10 * n)
    xc = center_x + radius * np.cos(eta)
    yc = center_y + radius * np.sin(eta)
    if TOWARDS_TURN:
        zc = np.full(eta.shape, l) - target_pitch * eta / (2 * np.pi)
    else:
        zc = target_pitch * eta / (2 * np.pi)

    p = [None] * (n + 2)
    q = [None] * (n + 2)
    v = [None] * (n + 2)
    u = [None] * (n + 2)
    for i in range(0, n):
        if draw:
            fig.add_trace(go.Scatter3d(
                x=[x[i], xp[i]],
                y=[y[i], yp[i]],
                z=[z[i], zp[i]],
                mode='lines',
                line=dict(color='red', width=4),
                name=f'Roller {i:d}'
            ))
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

    A = [None] * (2 * n + 1)
    b = [None] * (2 * n + 1)
    s = [None] * (2 * n + 2)
    if TOWARDS_TURN:
        fp = q[0]
        lp = p[0]
    else:
        fp = p[0]
        lp = q[0]
    s[0] = fp
    for i in range(0, 2 * n + 1):
        ri = i % n
        A[i] = np.zeros([7, 7])
        b[i] = np.zeros(7)

        A[i][0, 0] = 1
        A[i][0, 3] = -1
        b[i][0] = s[i][0]
        A[i][1, 1] = 1
        A[i][1, 4] = -1
        b[i][1] = s[i][1]
        A[i][2, 2] = 1
        A[i][2, 5] = -1
        b[i][2] = s[i][2]

                        
        A[i][3, 3] = u[ri + 1][0]
        A[i][3, 4] = u[ri + 1][1]
        A[i][3, 5] = u[ri + 1][2]

        A[i][4, 0] = 1
        A[i][4, 6] = -u[ri + 1][0]
        b[i][4] = p[ri + 1][0]
        A[i][5, 1] = 1
        A[i][5, 6] = -u[ri + 1][1]
        b[i][5] = p[ri + 1][1]
        A[i][6, 2] = 1
        A[i][6, 6] = -u[ri + 1][2]
        b[i][6] = p[ri + 1][2]

        # print(i)
        # print(A[i])
        # print(b[i])
        
        sol = np.linalg.solve(A[i], b[i])
        snxt = np.array([sol[0], sol[1], sol[2]]) 
        d = np.array([sol[3], sol[4], sol[5]])
        t = sol[6]
        s1 = snxt
        s2 = s[i] + d
        s3 = p[ri + 1] + t * u[ri + 1]

        dist1 = np.linalg.norm(s1 - s2)
        dist2 = np.linalg.norm(s2 - s3)
        dist3 = np.linalg.norm(s3 - s1)

        # print(f"dist1 = {dist1}, dist2 = {dist2}, dist3 = {dist3}")
        assert(dist1 <= TOL)
        assert(dist2 <= TOL)
        assert(dist3 <= TOL)

        s[i + 1] = snxt
        if draw:
            fig.add_trace(go.Scatter3d(
                x=[s[i][0], s[i + 1][0]],
                y=[s[i][1], s[i + 1][1]],
                z=[s[i][2], s[i + 1][2]],
                mode='lines',
                line=dict(color='magenta', width=4),
                name=f'Segment Line {i:d}'
            ))

    r = fp.copy()
    r[2] = lp[2]
    if draw:
        fig.add_trace(go.Scatter3d(
            x=[fp[0], r[0]],
            y=[fp[1], r[1]],
            z=[fp[2], r[2]],
            mode='lines',
            line=dict(color='green', width=4),
            name='Top Line'
        ))
        fig.add_trace(go.Scatter3d(
            x=[fp[0], r[0]],
            y=[fp[1], r[1]],
            z=[fp[2], target_pitch],
            mode='lines',
            line=dict(color='black', width=4),
            name='Target Pitch'
        ))
        fig.add_trace(go.Scatter3d(
            x = xc,
            y = yc,
            z = zc,
            mode='lines',
            line=dict(color='purple', width=4),
            name='Coil'
        ))
    [st, pt] = closest_points_between_segments(s[n], s[n + 1], fp, r)

    assert(abs(pt[0] - fp[0]) <= TOL)
    assert(abs(pt[1] - fp[1]) <= TOL)
    pitch_3d = np.linalg.norm(pt - fp)
    pitch_2d = pt[2] - fp[2]    
    assert(abs(pitch_3d - pitch_2d) <= TOL)
    simulated_pitch = pitch_3d

    diff = simulated_pitch - target_pitch
    print(f"degrees_clockwise = {degrees_clockwise}, simulated_pitch = {simulated_pitch}, DIFF = {diff}")

    if draw:
        # Camera configuration matching elev=-15, azim=20, roll=90
        camera = dict(
            # Camera position in 3D space
            eye=dict(x=1.32, y=0.48, z=-0.38),
            # Rotates the camera view around its axis (Roll = 90 deg)
            up=dict(x=-0.34, y=0.94, z=0.0)
        )
        fig.update_layout(
            title=f'Degrees clockwise = {degrees_clockwise:.4f}, Pitch error = {diff:.4f}',
            scene=dict(
                xaxis_title="X Axis",
                yaxis_title="Y Axis",
                zaxis_title="Z Axis",
                aspectmode="data",  # Ensures 1:1:1 scale ratio, matching Matplotlib's default
                camera=camera
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )

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