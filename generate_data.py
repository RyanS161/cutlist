import numpy as np
from design import (
    visualize,
    WoodDesign,
    ArbitraryCuboid,
    LibraryPrimitive,
    FootprintPrimitive,
)
from extern_datasets import get_partnet_sample, get_brickgpt_data
import pyvista as pv
from tqdm import tqdm
from itertools import permutations
import copy
import argparse
import os
import pandas as pd
from joblib import Parallel, delayed
import json
from sklearn.model_selection import train_test_split

# Configuration values
BOUNDS_DIM_X = 400
BOUNDS_DIM_Y = 400
BOUNDS_DIM_Z = 500

BOUNDS_CENTER_X = BOUNDS_DIM_X / 2
BOUNDS_CENTER_Y = BOUNDS_DIM_Y / 2

### Functions for fitting primitives to meshes


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


### Functions for scoring mesh fit


def score_cuboid_fit(lengths, points, transform):
    # Align the points to the cuboid axes
    point_centroid = np.mean(points, axis=0)
    centered_points = points - point_centroid
    aligned_points = centered_points @ transform[:3, :3].T

    # compute the difference between the bounding box of the aligned points and the cuboid lengths
    bounding_box = aligned_points.max(axis=0) - aligned_points.min(axis=0)
    score = np.linalg.norm(bounding_box - np.array(lengths))
    return score


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
    # visualize([original_voxels, fitted_voxels], colors=["red", "blue"], opacities=[0.8, 0.8], axis_length=25, show_image=True)

    return score


### Random utility functions


def get_rotation_matrix(axis, theta):
    ct, st = np.cos(theta), np.sin(theta)

    if axis == 0:  # x-axis rotation
        return np.array([[1, 0, 0], [0, ct, -st], [0, st, ct]])
    elif axis == 1:  # y-axis rotation
        return np.array([[ct, 0, st], [0, 1, 0], [-st, 0, ct]])
    else:  # z-axis rotation
        return np.array([[ct, -st, 0], [st, ct, 0], [0, 0, 1]])


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
    return rotation_matrix


### Different fitting strategies


def arbitrary_cuboids_strategy(meshes):
    fitted_parts = []
    for mesh in meshes:
        # Fit a primitive shape to the mesh
        transform, bounds = fit_cuboid_to_points(mesh.points)
        fitted_part = ArbitraryCuboid(bounds, transform)
        #### do comparison to see the best mesh
        fitted_parts.append(fitted_part)
    return WoodDesign(fitted_parts, ArbitraryCuboid)


def fit_with_strategy(meshes, strategy):
    resulting_parts = []
    for mesh in meshes:
        point_cloud = mesh.points
        best_part = strategy(point_cloud)
        resulting_parts.append(best_part.get_mesh())

    return resulting_parts


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

    return best_part


def fit_library_primitive(point_cloud):
    transform, bounds = fit_cuboid_to_points(point_cloud)
    best_part, best_mesh_score = None, np.inf
    for part_id, part_dims in LibraryPrimitive.PART_LIBRARY.items():
        part_lengths = np.array(part_dims)
        indices = find_closest_lengths_fit(bounds, part_lengths)
        extra_rotation = desired_rotation_from_axis_order(indices)
        new_transform = transform.copy()
        new_transform[:3, :3] = transform[:3, :3] @ extra_rotation
        mesh_score = score_cuboid_fit(part_lengths, point_cloud, new_transform)

        if mesh_score < best_mesh_score:
            best_mesh_score = mesh_score
            best_part = LibraryPrimitive(part_id=part_id, transform=new_transform)

    return best_part


### Functions for searching over part hierarchies and scales


def search_over_part_hierarchy(sample, strategy, scale=1.0, use_hierarchy=True):
    VOXEL_SIZE = 5 * scale
    meshes = sample["meshes"]
    # without hierarchy approach

    if not use_hierarchy:
        result_meshes = fit_with_strategy(meshes.values(), strategy)
        merged_mesh = pv.merge(
            [
                part.get_mesh()
                for part in arbitrary_cuboids_strategy(list(meshes.values())).parts
            ]
        )
        merged_parts = pv.merge(result_meshes)
        result_score = voxelized_iou_score(
            merged_mesh, merged_parts, voxel_size=VOXEL_SIZE
        )
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

        root_result_meshes = fit_with_strategy([merged_mesh], strategy)
        root_score = voxelized_iou_score(
            merged_mesh, root_result_meshes[0], voxel_size=VOXEL_SIZE
        )

        if ignore or children_score > root_score:
            return children_parts, children_score
        else:
            return root_result_meshes, root_score

    result_parts, result_score = iterate_nodes(part_tree, part_tree.root, ignore=True)

    return result_parts, result_score


# TODO: maybe we can try a hierarchical approach - first fit with coarse scales, then refine around the best scale
def search_over_scales(
    sample,
    strategy,
    scales=[
        1,
    ],
    use_hierarchy=True,
    filename="test",  # TODO: Remove this
):
    meshes = sample["meshes"]

    best_scale_score = np.inf
    best_scale_meshes = None
    best_scale = None
    for scale in tqdm(scales):
        scaled_sample = copy.deepcopy(sample)
        scaled_sample["meshes"] = {
            k: mesh.scale(scale, point=(BOUNDS_CENTER_X, BOUNDS_CENTER_Y, 0))
            for k, mesh in meshes.items()
        }
        result_meshes, result_score = search_over_part_hierarchy(
            scaled_sample, strategy, scale=scale, use_hierarchy=use_hierarchy
        )

        # pc = pv.PolyData(
        #     np.vstack([mesh.points[::10] for mesh in scaled_sample["meshes"].values()])
        # )
        # visualize(
        #     [pc] + result_meshes,
        #     colors=["red"] + ["tan"] * len(result_meshes),
        #     filename=f"{filename}_score{result_score * 100:.0f}_scale{scale * 10:.0f}.png",
        #     axis_length=100,
        #     text=f"Scale: {scale: .4f}, Score: {result_score:.4f}",
        # )
        if result_score < best_scale_score:
            best_scale_score = result_score
            best_scale_meshes = result_meshes
            best_scale = scale

    print(f"Best scale: {best_scale}")

    return best_scale_meshes, best_scale


### Functions for data generation and merging


def generate_design_data(partnet_dir, brickgpt_dir, output=None, n_jobs=-1):
    # partnet_dir is the directory where partnet data is stored, with each model being a subfolder with a numeric name
    # brickgpt_dir is the directory where brickgpt data is stored
    # n_jobs: number of parallel jobs (-1 uses all available cores)

    desired_models = os.listdir(partnet_dir)
    desired_models = list(filter(lambda x: x.isnumeric(), desired_models))

    print("Getting brickgpt data")
    lego_gpt_df = get_brickgpt_data(dir=brickgpt_dir)
    lego_gpt_df = lego_gpt_df[["object_id", "captions"]]
    lego_gpt_df = lego_gpt_df.drop_duplicates(subset=["object_id"])
    valid_object_ids = set(lego_gpt_df["object_id"].values)

    def process_single_model(model, partnet_dir, valid_object_ids):
        """Process a single model and return its design text if valid."""
        model_dir = os.path.join(partnet_dir, model)
        sample = get_partnet_sample(
            model_dir, max_parts=100, desired_xy=(BOUNDS_CENTER_X, BOUNDS_CENTER_Y)
        )

        if sample is None:
            return None

        model_id = sample["model_id"]
        if model_id not in valid_object_ids:
            return None

        original_meshes = sample["meshes"]
        arbitrary_design = arbitrary_cuboids_strategy(original_meshes.values())
        arbitrary_design_text = arbitrary_design.to_txt()

        return (model_id, arbitrary_design_text)

    print(
        f"Starting partnet processing with {n_jobs if n_jobs > 0 else 'all available'} cores"
    )

    # Process models in parallel with progress bar
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_single_model)(model, partnet_dir, valid_object_ids)
        for model in tqdm(desired_models, desc="Processing models")
    )

    # Filter out None results and convert to dictionary
    partnet_models = {
        model_id: design_txt
        for result in results
        if result is not None
        for model_id, design_txt in [result]
    }

    print(f"Successfully processed {len(partnet_models)} models")

    partnet_df = pd.DataFrame.from_dict(
        partnet_models, orient="index", columns=["design_txt"]
    )
    partnet_df.index.name = "object_id"
    merged_df = pd.merge(partnet_df, lego_gpt_df, on=["object_id"], how="left")

    if output:
        merged_df.to_csv(output, index=False)

    return merged_df


def create_instruction(caption):
    instruction = (
        "Create a wooden model of the input. Format your response as a list of wooden pieces: "
        "<dimensions> <position> <rotation>, where piece position is x y z and piece rotation is rx ry rz. "
        "All values are space separated, and pieces are separated with a newline."
        "No negative values are allowed. Rotations are in degrees from 0 to 179."
        "\n\n"
        "### Input:\n"
        f"{caption}"
    )
    return instruction


def generate_finetuning_data(data_df, output_dir=None):
    # Generating fine-tuning data

    finetuning_entries = []
    for index, row in data_df.iterrows():
        design_txt = row["design_txt"]

        if isinstance(row["captions"], str):
            # parse the captions as an array of strings
            captions = row["captions"].strip("[]").split("'\n '")
        else:
            captions = row["captions"]

        for caption in captions:
            caption = caption.strip().strip("'").strip()
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": create_instruction(caption)},
                {"role": "assistant", "content": design_txt},
            ]
            finetuning_entry = {"messages": messages}

            finetuning_entries.append(finetuning_entry)

    # Split into train, val, test
    train_entries, test_entries = train_test_split(
        finetuning_entries, test_size=0.1, random_state=42
    )

    for split_name, split_entries in zip(
        ["train", "test"], [train_entries, test_entries]
    ):
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            split_output_path = os.path.join(output_dir, f"{split_name}.jsonl")
            with open(split_output_path, "w") as f:
                for entry in split_entries:
                    f.write(json.dumps(entry) + "\n")


### Functions for testing


def local_test():
    desired_models = ["1299", "10027", "18258", "18740", "44970"]

    # Need to investigate how to integrate the part hierarchy as well
    for num in desired_models:
        sample = get_partnet_sample(
            f"/Users/ryanslocum/Documents/current_courses/semesterProject/cutlist/scratch/{num}",
            desired_xy=(BOUNDS_CENTER_X, BOUNDS_CENTER_Y),
        )
        original_meshes = sample["meshes"]
        original_point_cloud = pv.PolyData(
            np.vstack([mesh.points[::10] for mesh in original_meshes.values()])
        )
        show_image = False

        arbitrary_design = arbitrary_cuboids_strategy(original_meshes.values())
        arbitrary_design_text = arbitrary_design.to_txt()

        post_text_design = WoodDesign.from_txt(arbitrary_design_text, ArbitraryCuboid)
        arbitrary_meshes = [part.get_mesh() for part in post_text_design.parts]
        visualize(
            [original_point_cloud] + arbitrary_meshes,
            colors=["red"] + ["tan"] * len(arbitrary_meshes),
            filename=f"designs/arbitrary_fitted_meshes_{num}.png",
            show_image=show_image,
            axis_length=25,
            bounds=(0, BOUNDS_DIM_X, 0, BOUNDS_DIM_Y, 0, BOUNDS_DIM_Z),
        )

        search_scales = np.linspace(1.0, 5.0, 25)
        use_hierarchy = True
        length_meshes, best_scale = search_over_scales(
            sample,
            fit_footprint_primitive,
            scales=search_scales,
            use_hierarchy=use_hierarchy,
            filename=f"designs/length_fitted_meshes_{num}",
        )
        visualize(
            [original_point_cloud] + length_meshes,
            colors=["red"] + ["tan"] * len(length_meshes),
            filename=f"designs/length_fitted_meshes_{num}.png",
            show_image=show_image,
            axis_length=25,
            bounds=(0, BOUNDS_DIM_X, 0, BOUNDS_DIM_Y, 0, BOUNDS_DIM_Z),
        )

        our_meshes, best_scale = search_over_scales(
            sample,
            fit_library_primitive,
            scales=search_scales,
            use_hierarchy=use_hierarchy,
            filename=f"designs/our_fitted_meshes_{num}",
        )
        visualize(
            [original_point_cloud] + our_meshes,
            colors=["red"] + ["tan"] * len(our_meshes),
            filename=f"designs/our_fitted_meshes_{num}.png",
            show_image=show_image,
            axis_length=25,
            bounds=(0, BOUNDS_DIM_X, 0, BOUNDS_DIM_Y, 0, BOUNDS_DIM_Z),
        )


# Get a random model from the finetuning data and visualize it
def visualize_random_finetuning_model(finetuning_data_path):
    with open(finetuning_data_path, "r") as f:
        lines = f.readlines()
        random_line = np.random.choice(lines)
        entry = json.loads(random_line)
        messages = entry["messages"]
        design_txt = messages[-1]["content"]
        design = WoodDesign.from_txt(design_txt, ArbitraryCuboid)
        meshes = [part.get_mesh() for part in design.parts]
        visualize(
            meshes,
            colors=["tan"] * len(meshes),
            filename="./random_finetuning_model.png",
            show_image=True,
        )


if __name__ == "__main__":
    # local_test()
    # exit()

    # Process args for input directory name:
    parser = argparse.ArgumentParser(
        description="Generate WoodDesign data from PartNet and BrickGPT datasets"
    )
    parser.add_argument(
        "--partnet_dir",
        type=str,
        required=True,
        help="Path to the PartNet data directory",
    )
    parser.add_argument(
        "--brickgpt_dir",
        type=str,
        required=True,
        help="Path to the BrickGPT data directory",
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to save the generated CSV data"
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs to use (-1 for all available cores, default: -1)",
    )
    args = parser.parse_args()

    intermediate_csv = os.path.join(args.output, "intermediate.csv")
    data_df = generate_design_data(
        args.partnet_dir, args.brickgpt_dir, intermediate_csv, args.n_jobs
    )

    finetuning_output_dir = os.path.join(args.output, "finetuning_data")
    generate_finetuning_data(
        data_df,
        output_dir=finetuning_output_dir,
    )


# TODO: clean up this file into the right parts (and make new generate data file)

# TODO: Incorporate text from part-level annotations to help guide part generation??
