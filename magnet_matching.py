import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial.distance import cdist

# Parameters
n_points = 6
steps = 500
dt = 0.05
k_attract = 1.0
k_repel = 1.0
power_attract = 1
power_repel = 2
damping = 0.95
min_velocity_threshold = 1e-3
min_angular_velocity_threshold = 1e-3

# Generate point shape
np.random.seed(1)
shape = np.random.rand(n_points, 2) * 2

# Set A at center
A_center = np.array([6.0, 6.0])
A = shape - np.mean(shape, axis=0) + A_center

# Set B offset from A
B_center_offset = np.array([1.0, -1.0])
B = shape - np.mean(shape, axis=0) + A_center + B_center_offset

# Initial velocity and angular velocity of B
velocity = np.zeros(2)
angular_velocity = 0.0

def center_of_mass(points):
    return np.mean(points, axis=0)

def compute_forces_rigid(A, B):
    F_net = np.zeros(2)
    torque = 0.0
    B_cm = center_of_mass(B)

    for a in A:
        for b in B:
            d = a - b
            dist = np.linalg.norm(d) + 1e-6
            f = k_attract * d / (dist ** power_attract)
            r = b - B_cm
            F_net += f
            torque += np.cross(r, f)

    for i in range(len(B)):
        for j in range(len(B)):
            if i != j:
                d = B[i] - B[j]
                dist = np.linalg.norm(d) + 1e-6
                f = k_repel * d / (dist ** power_repel)
                r = B[i] - B_cm
                F_net += f / len(B)
                torque += np.cross(r, f)

    return F_net, torque

def rigid_transform(points, center, rotation, translation):
    R = np.array([[np.cos(rotation), -np.sin(rotation)],
                  [np.sin(rotation),  np.cos(rotation)]])
    return (points - center) @ R.T + center + translation

def closest_pairs(A, B):
    dist_matrix = cdist(A, B)
    pairs = []
    for i in range(len(A)):
        j = np.argmin(dist_matrix[i])
        pairs.append((A[i], B[j]))
    return pairs

# Plot setup
fig, ax = plt.subplots()
scatA = ax.scatter([], [], c='blue', label='Set A (+)')
scatB = ax.scatter([], [], c='red', label='Set B (-)')
lines = [ax.plot([], [], 'k:', lw=1)[0] for _ in range(n_points)]

ax.set_xlim(0, 12)
ax.set_ylim(0, 12)
ax.set_aspect("equal")
ax.legend()
ax.set_title("Rigid Body Magnetic Matching (A fixed, B moves)")

def init():
    scatA.set_offsets(A)
    scatB.set_offsets(B)
    for line in lines:
        line.set_data([], [])
    return [scatA, scatB] + lines

B_center = center_of_mass(B)
B_angle = 0.0

def update(frame):
    global B, B_center, B_angle, velocity, angular_velocity

    F_net, torque = compute_forces_rigid(A, B)
    acceleration = F_net / len(B)
    angular_acc = torque / len(B)

    velocity = damping * (velocity + dt * acceleration)
    angular_velocity = damping * (angular_velocity + dt * angular_acc)

    delta_pos = dt * velocity
    delta_angle = dt * angular_velocity

    B_center += delta_pos
    B_angle += delta_angle

    B[:] = rigid_transform(B, center_of_mass(B), delta_angle, delta_pos)

    scatB.set_offsets(B)
    pairs = closest_pairs(A, B)
    for i, (a, b) in enumerate(pairs):
        lines[i].set_data([a[0], b[0]], [a[1], b[1]])

    if np.linalg.norm(velocity) < min_velocity_threshold and abs(angular_velocity) < min_angular_velocity_threshold:
        ani.event_source.stop()

    return [scatA, scatB] + lines

ani = animation.FuncAnimation(fig, update, init_func=init, frames=steps, blit=True, repeat=False)
plt.show()
ani.save("magnet_simulation.gif", writer="pillow", fps=30)
