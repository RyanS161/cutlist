import numpy as np
from design import (
    visualize,
    ArbitraryCuboid,
    WoodDesign,
    LibraryPrimitive,
    FootprintPrimitive,
)
from shapeNetHelper import get_partnet_sample
import pyvista as pv
from tqdm import tqdm
from itertools import permutations
import copy

# def visualize_txt_designs(designs):
#     EXAMPLE_DESIGNS = {
#         "X": "3 0 0 0 0 0 -45\n3 0 0 0 0 0 45",
#         "T": "5 0 0 0 0 0 0\n5 0 0 0 0 0 90",
#         "H": "0 -40 0 0 0 0 0\n0  40 0 0 0 0 0\n2   0 0 0 0 0 90",
#         "Chair": "0 0 0 0 0 0 0\n0 60 0 0 0 0 0\n0 0 60 0 0 0 0\n0 60 60 0 0 0 0\n4 30 20 30 0 0 0\n3 30 20 120 0 0 0\n",
#         "House": "0 0 0 60 0 0 0\n0 160 0 60 0 0 0\n0 0 80 60 0 0 0\n0 160 80 60 0 0 0\n6 80 40 5 0 0 0\n6 80 40 125 0 0 0\n5 80 0 65 90 0 0\n5 80 80 65 90 0 0\n4 40 40 65 90 90 0\n4 120 40 65 90 90 0\n8 80 0 65 0 0 0\n",
#         "Box": "0 10 10 5 0 0 0\n0 150 10 5 0 0 0\n0 10 70 5 0 0 0\n0 150 70 5 0 0 0\n5 80 10 2.5 0 0 0\n5 80 70 2.5 0 0 0\n2 10 40 60 0 0 0\n2 150 40 60 0 0 0\n6 80 40 117.5 90 0 0\n6 80 40 2.5 90 0 0\n",
#     }
#     for letter, design_txt in designs.items():
#         design = Design(
#             assembledComponents=[],
#             bounds=np.array([[-200, 200], [-200, 200], [0, 200]]),
#         )
#         design.from_txt(design_txt, from_file=False)
#         design.visualize_design(filename=f"designs/design_{letter}")


def score_points_to_cylinder(points, cylinder_axis, cylinder_radius, cylinder_height):
    distance_from_axis = np.linalg.norm(
        points - (points @ cylinder_axis[:, None]) * cylinder_axis[:, None].T, axis=1
    )

    half_height = cylinder_height / 2
    projections = np.abs(points @ cylinder_axis)

    inside_of_cylinder = (distance_from_axis <= cylinder_radius) & (
        projections <= half_height
    )

    above_cap = (distance_from_axis <= cylinder_radius) & (projections > half_height)

    # set distances for points inside the cylinder to the minimum of distance to side or cap
    inside_distances = np.minimum(
        cylinder_radius - distance_from_axis, half_height - projections
    )
    above_cap_distances = np.abs(projections - half_height)
    outside_distances = np.sqrt(
        (distance_from_axis - cylinder_radius) ** 2
        + np.maximum(0, projections - half_height) ** 2
    )

    distances = np.zeros(points.shape[0])
    distances[inside_of_cylinder] = inside_distances[inside_of_cylinder]
    distances[above_cap] = above_cap_distances[above_cap]
    distances[~(inside_of_cylinder | above_cap)] = outside_distances[
        ~(inside_of_cylinder | above_cap)
    ]

    return np.linalg.norm(distances)


def score_points_to_box(points, box_dims):
    # Idea one - minimum distance to box surface
    # Problem: boxes can be arbitrarily long
    # half_extents = np.array([box_x/2, box_y/2, box_z/2])
    # d = np.abs(points) - half_extents
    # outside = np.linalg.norm(np.maximum(d, 0), axis=1)
    # inside = np.minimum(np.max(d, axis=1), 0)
    # return np.linalg.norm(outside + np.abs(inside))

    # Idea two - distance to vertices
    # Penalizes larger shapes because they have vertices further away
    # box_vertices = np.array([
    #     [box_x / 2, box_y / 2, box_z / 2],
    #     [box_x / 2, box_y / 2, -box_z / 2],
    #     [box_x / 2, -box_y / 2, box_z / 2],
    #     [box_x / 2, -box_y / 2, -box_z / 2],
    #     [-box_x / 2, box_y / 2, box_z / 2],
    #     [-box_x / 2, box_y / 2, -box_z / 2],
    #     [-box_x / 2, -box_y / 2, box_z / 2],
    #     [-box_x / 2, -box_y / 2, -box_z / 2],
    # ])

    # # calculate the distance from each point to each vertex
    # distances = np.linalg.norm(points[:, None, :] - box_vertices[None, :, :], axis=2)
    # distances = np.min(distances, axis=1)
    # return np.mean(distances.flatten())

    # Idea three - compare the dimensions with the dimensions of the bounding box of the points

    bounding_box = points.max(axis=0) - points.min(axis=0)

    return np.linalg.norm(bounding_box - np.array(box_dims))


def voxelized_iou_score(original_mesh, fitted_mesh, voxel_size=1):
    # We may not have to do the voxelization operation if we know that the meshes are compositions of cuboids
    original_voxels = original_mesh.voxelize(spacing=voxel_size)
    fitted_voxels = fitted_mesh.voxelize(spacing=voxel_size)

    original_points = original_voxels.points
    fitted_points = fitted_voxels.points

    original_points_int = (original_points / voxel_size).astype(int)
    fitted_points_int = (fitted_points / voxel_size).astype(int)

    original_tuples = np.unique(
        original_points_int.view(
            np.dtype(
                (
                    np.void,
                    original_points_int.dtype.itemsize * original_points_int.shape[1],
                )
            )
        )
    )
    fitted_tuples = np.unique(
        fitted_points_int.view(
            np.dtype(
                (np.void, fitted_points_int.dtype.itemsize * fitted_points_int.shape[1])
            )
        )
    )

    intersection = len(np.intersect1d(original_tuples, fitted_tuples))
    union = len(np.union1d(original_tuples, fitted_tuples))

    if union == 0:
        score = 0.0
    else:
        score = intersection / union

    # visualize point clouds:
    # print(score)
    # visualize([original_voxels, fitted_voxels], colors=["red", "blue"], opacities=[0.8, 0.8], axis_length=25, off_screen=False)

    return score


def arbitrary_cuboids_strategy(meshes):
    fitted_parts = []
    for mesh in meshes:
        # Fit a primitive shape to the mesh
        transform, bounds = fit_cuboid_to_points(mesh.points)
        fitted_part = ArbitraryCuboid(bounds, transform)
        #### do comparison to see the best mesh
        fitted_parts.append(fitted_part)
    return WoodDesign(fitted_parts)


def fit_footprint_primitive(point_cloud):
    transform, bounds = fit_cuboid_to_points(point_cloud)
    best_part, best_mesh_score = None, np.inf
    for footprint_id, footprint in FootprintPrimitive.FOOTPRINTS.items():
        indices = find_closest_lengths_fit(bounds, footprint)
        extra_rotation = desired_rotation_from_axis_order(indices)
        new_transform = transform.copy()
        new_transform[:3, :3] = transform[:3, :3] @ extra_rotation
        part_lengths = np.array([footprint[0], footprint[1], bounds[indices[2]]])
        mesh_score = score_cuboid_fit(part_lengths, point_cloud, new_transform)

        if mesh_score < best_mesh_score:
            best_mesh_score = mesh_score
            best_part = FootprintPrimitive(
                part_id=footprint_id, length=part_lengths[2], transform=new_transform
            )

    return best_part, best_mesh_score


def fit_our_primitive(point_cloud):
    transform, bounds = fit_cuboid_to_points(point_cloud)
    best_part, best_mesh_score = None, np.inf
    for part_id, part_dims in LibraryPrimitive.PART_LIBRARY.items():
        part_lengths = np.array(part_dims)
        indices = find_closest_lengths_fit(bounds, part_lengths)
        extra_rotation = desired_rotation_from_axis_order(indices)
        new_transform = transform.copy()
        new_transform[:3, :3] = (
            transform[:3, :3] @ extra_rotation
        )  # Is this the right transform?

        # visualize the fitted part
        # fixed_part = LibraryPrimitive(part_id=part_id, transform=new_transform)
        # old_part = LibraryPrimitive(part_id=part_id, transform=transform)
        # visualize([fixed_part.get_mesh(), old_part.get_mesh(), pv.PolyData(point_cloud)], colors=["tan", "red", "blue"], opacities=[0.8, 0.8, 0.8], axis_length=25, off_screen=False)

        mesh_score = score_cuboid_fit(part_lengths, point_cloud, new_transform)

        if mesh_score < best_mesh_score:
            best_mesh_score = mesh_score
            best_part = LibraryPrimitive(part_id=part_id, transform=new_transform)
    return best_part, best_mesh_score


def search_over_part_hierarchy(sample, strategy, scale=1.0):
    VOXEL_SIZE = 5 / scale
    USE_HIERARCHY = False
    meshes = sample["meshes"]
    # without hierarchy approach

    if not USE_HIERARCHY:
        result_meshes, result_score = fit_and_score(meshes.values(), strategy)
        return result_meshes, result_score

    # with hierarchy approach
    part_tree = sample["part_tree"]

    def iterate_nodes(part_tree, node_id, ignore=False):
        node_meshes = part_tree[node_id].data
        # merge meshes into one:
        # merged_mesh = pv.merge([meshes[mesh_id] for mesh_id in node_meshes])
        original_meshes = [meshes[mesh_id] for mesh_id in node_meshes]
        merged_mesh = pv.merge(
            [
                part.get_mesh()
                for part in arbitrary_cuboids_strategy(original_meshes).parts
            ]
        )

        if not part_tree[node_id].is_leaf():
            children_parts = []
            for child_node in part_tree.children(node_id):
                child_result_parts, _ = iterate_nodes(part_tree, child_node.identifier)
                children_parts.extend(child_result_parts)
            # result_score += child_result_score

            # print("Root tag:", part_tree[node_id].tag)
            merged_child_parts = pv.merge([part for part in children_parts])
            children_score = voxelized_iou_score(
                merged_mesh, merged_child_parts, voxel_size=VOXEL_SIZE
            )
        else:
            children_score = 0  # if leaf, no children to consider

        root_result_meshes, _ = fit_and_score([merged_mesh], strategy)
        root_score = voxelized_iou_score(
            merged_mesh, root_result_meshes[0], voxel_size=VOXEL_SIZE
        )
        # if not part_tree[node_id].is_leaf():
        #     print(f"Root score: {root_score} child score: {children_score} for node [{part_tree[node_id].tag}] with children {[child.tag for child in part_tree.children(node_id)]}")
        #     visualize([merged_mesh, root_result_meshes[0], merged_child_parts], colors=["blue", "red", "yellow"], opacities=[0.8, 0.8, 0.8], axis_length=25, off_screen=False)

        if ignore or children_score > root_score:
            # print(f"Using children fit with score {children_score} over parent score {root_score}")
            return children_parts, children_score
        else:
            # print(f"Using parent fit with score {root_score} over children score {children_score}")
            return root_result_meshes, root_score

    result_parts, result_score = iterate_nodes(part_tree, part_tree.root, ignore=True)

    return result_parts, result_score


def fit_and_score(meshes, strategy):
    resulting_parts = []
    result_score = 0
    for mesh in meshes:
        point_cloud = mesh.points
        best_part, best_mesh_score = strategy(point_cloud)
        result_score += best_mesh_score
        resulting_parts.append(best_part.get_mesh())

    # Should change this to a single score: merge parts, merge point clouds, voxelize and compute IoU
    # result_score = voxelized_iou_score(result_meshes, meshes.values())

    return resulting_parts, result_score


# TODO: maybe we can try a hierarchical approach - first fit with coarse scales, then refine around the best scale
def search_over_scales(sample, strategy):
    meshes = sample["meshes"]
    scales = np.linspace(0.5, 4.0, 20)
    # scales = [1.0]

    best_scale_score = np.inf
    best_scale_meshes = None
    best_scale = None
    for scale in tqdm(scales, desc="Searching over scales"):
        scaled_sample = copy.deepcopy(sample)
        scaled_sample["meshes"] = {k: mesh.scale(scale) for k, mesh in meshes.items()}
        result_meshes, result_score = search_over_part_hierarchy(
            scaled_sample, strategy
        )
        if result_score < best_scale_score:
            best_scale_score = result_score
            best_scale_meshes = result_meshes
            best_scale = scale

    print(f"Best scale: {best_scale}")

    return best_scale_meshes, best_scale


def score_cuboid_fit(lengths, points, transform):
    point_centroid = np.mean(points, axis=0)
    centered_points = points - point_centroid
    aligned_points = centered_points @ transform[:3, :3].T

    score = score_points_to_box(aligned_points, lengths)

    return score


def find_two_closest_lengths(lengths_list):
    # Find the two lengths that are closest to each other (closest to a square cross-section)
    min_diff = np.inf
    best_pair = (None, None)
    for i in range(len(lengths_list)):
        for j in range(i + 1, len(lengths_list)):
            diff = abs(lengths_list[i] - lengths_list[j])
            if diff < min_diff:
                min_diff = diff
                best_pair = (i, j)
    return best_pair


def find_closest_lengths_fit(lengths_list, target):
    # Given a list of 3 lengths and 2 or 3 target lengths, return the indices of the lengths that best match the target lengths
    assert len(lengths_list) == 3
    lengths = np.array(lengths_list)
    perms = np.asarray(list(permutations([0, 1, 2])))
    perms = perms[:, : len(target)]

    best_perm = None
    min_diff = np.inf
    for perm in perms:
        # compute the difference between the selected lengths and the target lengths
        selected_lengths = lengths[list(perm)]
        diff = np.linalg.norm(selected_lengths - np.array(target))
        if diff < min_diff:
            min_diff = diff
            best_perm = perm

    if len(best_perm) == 2:
        # append the remaining index
        remaining_index = list({0, 1, 2} - set(best_perm))[0]
        best_perm = np.array(list(best_perm) + [remaining_index])
    return best_perm


def desired_rotation_from_axis_order(axes):
    rotation_matrix = np.zeros((3, 3))
    for old_axis, new_axis in enumerate(axes):
        rotation_matrix[new_axis, old_axis] = 1
    if np.linalg.det(rotation_matrix) < 0:
        # if the determinant is -1, we have a reflection, so swap two axes
        rotation_matrix[0, :] *= -1

    # visualize the rotation
    # arrows = [
    #     pv.Arrow(start=(0, 0, 0), direction=(1,0,0), scale=25),
    #     pv.Arrow(start=(0, 0, 0), direction=(0,1,0), scale=25),
    #     pv.Arrow(start=(0, 0, 0), direction=(0,0,1), scale=25),
    # ]
    # rotated_arrows = [ arrow.rotate(rotation_matrix, point=(0,0,0), inplace=False) for arrow in arrows ]
    # visualize(rotated_arrows, colors=["red", "green", "blue"], opacities=[0.8, 0.8, 0.8], axis_length=10, off_screen=False)

    return rotation_matrix


# Approach: iteratively rotate around each axis to minimize bounding box volume
def fit_cuboid_to_points(points, coarse_steps=20, fine_steps=10):
    MAX_ANGLE = np.pi / 4
    pts = np.asarray(points).astype(float)
    center = pts.mean(axis=0)
    pts_c = pts - center

    best_vol = np.inf
    pts_r = pts_c.copy()
    R_total = np.eye(3)

    for axis in range(3):
        # Stage 1: Coarse search across full range
        coarse_thetas = np.linspace(-MAX_ANGLE, MAX_ANGLE, coarse_steps)
        best_coarse_theta = 0
        best_coarse_vol = np.inf

        for theta in coarse_thetas:
            R_search = get_rotation_matrix(axis, theta)
            pts_candidate = (R_search @ pts_r.T).T

            extents = pts_candidate.max(axis=0) - pts_candidate.min(axis=0)
            vol = extents[0] * extents[1] * extents[2]

            if vol < best_coarse_vol:
                best_coarse_vol = vol
                best_coarse_theta = theta

        # Stage 2: Fine search around best coarse result
        theta_range = (np.pi / 2) / coarse_steps  # Range around best coarse angle
        fine_start = max(-MAX_ANGLE, best_coarse_theta - theta_range)
        fine_end = min(MAX_ANGLE, best_coarse_theta + theta_range)
        fine_thetas = np.linspace(fine_start, fine_end, fine_steps)

        best_rot = np.eye(3)
        best_fine_vol = best_coarse_vol

        for theta in fine_thetas:
            R_search = get_rotation_matrix(axis, theta)
            pts_candidate = (R_search @ pts_r.T).T

            extents = pts_candidate.max(axis=0) - pts_candidate.min(axis=0)
            vol = extents[0] * extents[1] * extents[2]

            if vol < best_fine_vol:
                best_fine_vol = vol
                best_rot = R_search

        # Update for next axis iteration
        if best_fine_vol < best_vol:
            best_vol = best_fine_vol

        R_total = best_rot @ R_total
        pts_r = (R_total @ pts_c.T).T

    # Final alignment and bounds calculation
    aligned = (R_total @ pts_c.T).T
    x_length, y_length, z_length = aligned.max(axis=0) - aligned.min(axis=0)

    transform = np.eye(4)
    transform[:3, :3] = R_total.T
    transform[:3, 3] = center

    return transform, (x_length, y_length, z_length)


def get_rotation_matrix(axis, theta):
    ct, st = np.cos(theta), np.sin(theta)

    if axis == 0:  # x-axis rotation
        return np.array([[1, 0, 0], [0, ct, -st], [0, st, ct]])
    elif axis == 1:  # y-axis rotation
        return np.array([[ct, 0, st], [0, 1, 0], [-st, 0, ct]])
    else:  # z-axis rotation
        return np.array([[ct, -st, 0], [st, ct, 0], [0, 0, 1]])


if __name__ == "__main__":
    desired_models = ["1299", "10027", "18258", "18740", "44970"]

    # Need to investigate how to integrate the part hierarchy as well
    for num in desired_models:
        sample = get_partnet_sample(
            f"/Users/ryanslocum/Downloads/cutlist/scratch/{num}"
        )
        original_meshes = sample["meshes"]
        original_point_cloud = pv.PolyData(
            np.vstack([mesh.points[::10] for mesh in original_meshes.values()])
        )
        off_screen = True

        arbitrary_design = arbitrary_cuboids_strategy(original_meshes.values())
        arbitrary_meshes = [part.get_mesh() for part in arbitrary_design.parts]
        visualize(
            [original_point_cloud] + arbitrary_meshes,
            colors=["red"] + ["tan"] * len(arbitrary_meshes),
            filename=f"designs/arbitrary_fitted_meshes_{num}.png",
            axis_length=25,
            off_screen=off_screen,
        )

        length_meshes, best_scale = search_over_scales(sample, fit_footprint_primitive)
        visualize(
            [original_point_cloud] + length_meshes,
            colors=["red"] + ["tan"] * len(length_meshes),
            filename=f"designs/length_fitted_meshes_{num}.png",
            axis_length=25,
            off_screen=off_screen,
        )

        our_meshes, best_scale = search_over_scales(sample, fit_our_primitive)
        visualize(
            [original_point_cloud] + our_meshes,
            colors=["red"] + ["tan"] * len(our_meshes),
            filename=f"designs/our_fitted_meshes_{num}.png",
            axis_length=25,
            off_screen=off_screen,
        )
