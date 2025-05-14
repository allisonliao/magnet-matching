import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


# Kabsch algorithm program
def best_fit_transform(P, Q):
    """
    Returns the rotation matrix R and translation vector t that aligns P to Q
    using the Kabsch algorithm.
    """
    # center points
    P_centroid = np.mean(P, axis=0)
    Q_centroid = np.mean(Q, axis=0)
    P_centered = P - P_centroid
    Q_centered = Q - Q_centroid

    # compute covariance matrix
    H = P_centered.T @ Q_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # fix improper rotation
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = Q_centroid - R @ P_centroid
    return R, t


# Matching method
def compute_matching(A, B):
    dist_matrix = cdist(A, B)
    row_ind, col_ind = linear_sum_assignment(dist_matrix)
    return row_ind, col_ind


# Initial setup
np.random.seed(0)
n_points = 6
A = np.random.rand(n_points, 2) * 5
angle = np.pi / 4
rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
B = (
    A @ rotation.T + np.array([2, 1]) + np.random.normal(0, 0.3, size=A.shape)
)  # noisy rotated + translated version

# find best transformation (rigid -- translation, rotations)
R_opt, t_opt = best_fit_transform(B, A)


# Animation prepare
fig, ax = plt.subplots()
scatA = ax.scatter(A[:, 0], A[:, 1], c="blue", label="Set A (target)")
scatB = ax.scatter(B[:, 0], B[:, 1], c="red", label="Set B (original)")
scatBprime = ax.scatter([], [], c="green", label="Set B' (transformed)")
lines = [ax.plot([], [], "k--")[0] for _ in range(n_points)]

ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_aspect("equal")
ax.legend()
ax.set_title("Rigid Transformation Alignment (Kabsch)")

steps = 60


def init():
    scatBprime.set_offsets(np.zeros_like(B))
    for line in lines:
        line.set_data([], [])
    return [scatBprime] + lines


def update(frame):
    t = frame / (steps - 1)
    R_interp = (1 - t) * np.eye(2) + t * R_opt
    t_interp = t * t_opt
    B_transformed = (B @ R_interp.T) + t_interp

    scatBprime.set_offsets(B_transformed)

    # draw matching
    row_ind, col_ind = compute_matching(A, B_transformed)
    for i, (ai, bi) in enumerate(zip(row_ind, col_ind)):
        lines[i].set_data(
            [A[ai, 0], B_transformed[bi, 0]], [A[ai, 1], B_transformed[bi, 1]]
        )

    return [scatBprime] + lines


ani = animation.FuncAnimation(fig, update, init_func=init, frames=steps, blit=True)
plt.show()
