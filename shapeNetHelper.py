import binvox_rw
import numpy as np
from scipy.spatial.transform import Rotation as R
import pyvista as pv
import pandas as pd
import os
import json


def zero_shapenet_obj(mesh, target_size=100, return_transfrom=False):
    # Rotate mesh to stand upright
    mesh = mesh.rotate_x(90)
    # Center mesh at origin
    mesh_width = mesh.bounds[1] - mesh.bounds[0]
    mesh_depth = mesh.bounds[3] - mesh.bounds[2]
    # scale to be 100 millimeters wide on the smallest side
    scale = target_size / max(mesh_width, mesh_depth)
    mesh = mesh.scale([scale, scale, scale])
    z_height = mesh.bounds[5] - mesh.bounds[4]
    translation = np.array((0, 0, z_height / 2)) - np.array(mesh.center)
    mesh = mesh.translate(translation)

    if return_transfrom:
        return mesh, translation, scale
    else:
        return mesh


def get_shapenet_samples():
    samples = []

    SHAPENET_PARENT_DIR = "/Users/ryanslocum/Downloads/cutlist/ShapeNetCore"
    df = pd.read_csv(
        os.path.join(SHAPENET_PARENT_DIR, "shapenet_models_with_captions.csv"),
        dtype={"category_id": str},
    )
    # choose 5 random samples from each category
    sample_df = (
        df.groupby("category")
        .apply(lambda x: x.sample(5, random_state=100))
        .reset_index(drop=True)
    )
    for _, row in sample_df.iterrows():
        category = row["category"]
        category_id = row["category_id"]
        object_id = row["object_id"]
        obj_file = os.path.join(
            SHAPENET_PARENT_DIR,
            category_id,
            str(object_id),
            "models",
            "model_normalized.obj",
        )
        if os.path.isfile(obj_file):

            mesh = zero_shapenet_obj(pv.read(obj_file))
            point_cloud = None

            with open(obj_file.replace(".obj", ".surface.binvox"), "rb") as f:
                voxelized_model = binvox_rw.read_as_3d_array(f)
                voxel_data = voxelized_model.data
                # make point cloud from voxel data
                points = np.argwhere(voxel_data)
                point_cloud = pv.PolyData(points[::10])
                point_cloud = zero_shapenet_obj(point_cloud)

            sample = {
                "file": obj_file,
                "transformed_mesh": mesh,
                "point_cloud": point_cloud,
                "category": category,
                "object_id": object_id,
            }
            samples.append(sample)
        else:
            print(f"OBJ file not found: {obj_file}")

    return samples


def extract_nested_objs(data, is_top_level=True):
    """
    Recursively extract all 'objs' arrays from nested JSON structure,
    ignoring the top-level one.
    """
    all_objs = []

    if isinstance(data, dict):
        # If this dict has 'objs' and it's not the top level, collect them
        if "objs" in data and not is_top_level:
            all_objs.extend(data["objs"])

        # Recursively process all values in the dict
        for key, value in data.items():
            if key == "children" or isinstance(value, (list, dict)):
                all_objs.extend(extract_nested_objs(value, False))

    elif isinstance(data, list):
        # Process each item in the list
        for item in data:
            all_objs.extend(extract_nested_objs(item, False))

    return all_objs


def get_partnet_sample(dir):
    sample = {}
    sample["meshes"] = get_and_transform_partnet_meshes(dir)
    sample["category"] = json.load(open(os.path.join(dir, "meta.json"), "r")).get(
        "model_cat", ""
    )
    result_after_merging_json = json.load(
        open(os.path.join(dir, "result_after_merging.json"), "r")
    )
    part_combinations = extract_nested_objs(result_after_merging_json)
    sample["part_combinations"] = part_combinations
    return sample


def get_and_transform_partnet_meshes(dir):
    original_meshes = {}
    combined_mesh = pv.PolyData()
    for file in os.listdir(os.path.join(dir, "objs")):
        if file.endswith(".obj"):
            mesh_id = file.replace(".obj", "")
            mesh = pv.read(os.path.join(dir, "objs", file)).clean()
            original_meshes[mesh_id] = mesh
            combined_mesh = combined_mesh.merge(mesh)

    _, translation, scale = zero_shapenet_obj(combined_mesh, return_transfrom=True)

    transformed_meshes = {}
    for mesh_id, mesh in original_meshes.items():
        new_mesh = mesh.rotate_x(90)
        # scale to be 100 millimeters wide on the smallest side
        new_mesh = new_mesh.scale([scale, scale, scale])
        new_mesh = new_mesh.translate(translation)
        transformed_meshes[mesh_id] = new_mesh

    return transformed_meshes
