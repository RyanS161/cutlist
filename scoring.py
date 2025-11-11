from design import ArbitraryCuboid
from generate_data import BOUNDS_DIM_X, BOUNDS_DIM_Y
from scipy.spatial import HalfspaceIntersection, ConvexHull
from scipy.optimize import linprog
import numpy as np


def find_feasible_point(A, b, tol=1e-9):
    m, n = A.shape
    c = np.zeros(n)  # objective: minimize 0 (just find any feasible x)
    res = linprog(c, A_ub=A, b_ub=-b, method="highs")  # A x <= -b
    if res.success:
        return res.x  # feasible point (may be on boundary)
    return None  # infeasible


def find_strictly_interior_point(A, b, tol=1e-9):
    m, n = A.shape
    # Variables are (x[0], x[1], x[2], t)
    c = np.zeros(n + 1)
    c[-1] = -1.0  # maximize t -> minimize -t

    # Build constraint: A x + b <= -t  ->  A|1 * [x, t] <= -b
    A_lp = np.hstack([A, np.ones((m, 1))])
    b_lp = -b

    # Also require t >= 0
    bounds = [(None, None)] * n + [(0, None)]

    res = linprog(c, A_ub=A_lp, b_ub=b_lp, bounds=bounds, method="highs")

    if not res.success:
        return None, 0.0  # infeasible or no interior

    x = res.x[:n]
    t = res.x[-1]

    if t > tol:
        return x, t  # strictly interior point
    else:
        return None, t  # feasible but zero-volume (flat)


def get_halfspaces(cube: ArbitraryCuboid):
    R = cube.transform[:3, :3]  # rotation
    c = cube.transform[:3, 3]  # centroid
    L = np.asarray(cube.dims).reshape(3)
    A_rows = []
    b_rows = []
    for i in range(3):
        u = R[:, i]  # world direction of local +axis i
        h = L[i] / 2.0
        # + face: u^T x <= u^T c + h  ->  (u^T) x  + ( -u^T c - h ) <= 0
        A_rows.append(u)
        b_rows.append(-(u.dot(c) + h))
        # - face: -u^T x <= -u^T c + h  ->  (-u^T) x  + ( u^T c - h ) <= 0
        A_rows.append(-u)
        b_rows.append(u.dot(c) - h)
    A = np.vstack(A_rows)
    b = np.array(b_rows)

    return np.hstack([A, b.reshape(-1, 1)])


def cube_iou(cube1: ArbitraryCuboid, cube2: ArbitraryCuboid) -> float:
    halfspace1 = get_halfspaces(cube1)
    halfspace2 = get_halfspaces(cube2)
    all_halfspaces = np.vstack([halfspace1, halfspace2])

    feasible_point = find_feasible_point(all_halfspaces[:, :-1], all_halfspaces[:, -1])

    if feasible_point is None:
        return 0.0  # Cubes are not intersecting (no feasible point found).

    interior_point, t = find_strictly_interior_point(
        all_halfspaces[:, :-1], all_halfspaces[:, -1]
    )

    if interior_point is None:
        return 0.0  # Cubes are intersecting but form a zero-volume intersection.

    hs = HalfspaceIntersection(all_halfspaces, interior_point)
    # hs.intersections gives the vertices
    verts = hs.intersections
    hull = ConvexHull(verts)
    volume = hull.volume

    volume1 = np.prod(cube1.dims)
    volume2 = np.prod(cube2.dims)

    iou = volume / (volume1 + volume2 - volume)
    return iou


def obb_axes(R):
    """Return the 3 world-space axes (columns) from a 3x3 rotation matrix."""
    return [R[:, 0], R[:, 1], R[:, 2]]


def sat_project(box_center, axes, half_extents, axis):
    """Project an OBB onto a given axis, return [min, max] interval."""
    axis = axis / np.linalg.norm(axis)
    # projected center
    c = box_center @ axis
    # projected radius (sum of |hi * dot(ui, axis)|)
    r = sum(abs(h * (u @ axis)) for h, u in zip(half_extents, axes))
    return c - r, c + r


def interval_distance(i1, i2):
    """Gap between 1D intervals. Negative means overlap."""
    return max(i1[0] - i2[1], i2[0] - i1[1])


def obb_distance(cube1: ArbitraryCuboid, cube2: ArbitraryCuboid, eps=1e-12):
    """Return signed distance (>0 if separated, 0 if touching/intersecting)."""

    c1, L1, R1 = cube1.transform[:3, 3], np.array(cube1.dims), cube1.transform[:3, :3]
    c2, L2, R2 = cube2.transform[:3, 3], np.array(cube2.dims), cube2.transform[:3, :3]

    a1 = obb_axes(R1)
    a2 = obb_axes(R2)
    h1 = L1 / 2.0
    h2 = L2 / 2.0

    min_gap = -np.inf  # track maximum separating gap

    # candidate axes: 3 face normals of each, and 9 cross-product axes
    axes = a1 + a2 + [np.cross(u, v) for u in a1 for v in a2]

    for axis in axes:
        if np.linalg.norm(axis) < eps:  # skip degenerate cross product
            continue
        i1 = sat_project(c1, a1, h1, axis)
        i2 = sat_project(c2, a2, h2, axis)
        gap = interval_distance(i1, i2)
        if gap > 0:  # they are separated along this axis
            min_gap = max(min_gap, gap)
        else:
            # overlap on this axis -> contributes no positive distance
            min_gap = max(min_gap, 0)

    return min_gap


def assemblability_score(cube1: ArbitraryCuboid, cube2: ArbitraryCuboid) -> float:
    """Compute an assemblability score based on OBB distance."""
    # Positive score means separated, negative means overlapping. 0 means touching.
    TANH_SCALING_FACTOR = 1e-2
    distance = obb_distance(cube1, cube2)
    if distance == 0:
        iou = cube_iou(cube1, cube2)
        score = -iou
    else:
        score = np.tanh(distance * TANH_SCALING_FACTOR)
    return score


def score_new_part(design, new_part) -> float:
    """Compute the minimum assemblability score of new_part with all existing parts."""
    # TODO: Need to program in bounds from the design later
    FLOOR_DEPTH = 1
    floor_transform = np.eye(4)
    floor_transform[:3, 3] = np.array([BOUNDS_DIM_X, BOUNDS_DIM_Y, -FLOOR_DEPTH]) / 2.0
    FLOOR_PART = ArbitraryCuboid(
        dims=[BOUNDS_DIM_Y, 1000, FLOOR_DEPTH], transform=floor_transform
    )
    scores = np.zeros(len(design.parts) + 1)
    scores[-1] = assemblability_score(new_part, FLOOR_PART)
    for i, existing_part in enumerate(design.parts):
        scores[i] = assemblability_score(new_part, existing_part)

    # If there is a negative score (overlap), return the minimum (most negative) score
    # Else, return the minimum positive score (closest to zero)
    # Also get the scored part
    scored_idx = np.argmin(scores)
    score = scores[scored_idx]

    if scored_idx == len(design.parts):
        # Scored against floor
        return score, None

    return score, scored_idx
