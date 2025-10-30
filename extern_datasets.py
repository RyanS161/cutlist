import numpy as np
import pyvista as pv
import pandas as pd
import os
import json
import treelib


"""ShapeNet Helpers:"""


def offset_to_word(offset, pos="n"):
    from nltk.corpus import wordnet as wn

    # offset: string or int, e.g. '02691156'
    # pos: part of speech, 'n' for noun (most ShapeNet classes are nouns)
    synset = wn.synset_from_pos_and_offset(pos, int(offset))
    return synset.name(), synset.definition()


def collect_shapenet_data(dir="/Users/ryanslocum/Downloads/cutlist/ShapeNetCore"):
    shapenet_data = []
    for category_id in os.listdir(dir):
        category_folder = os.path.join(dir, category_id)
        if os.path.isdir(category_folder):
            if not category_id.isdigit():
                continue
            category = offset_to_word(category_id)[0]
            print(f"Entering Category: {category}, Offset: {category_id}")
            for object_id in os.listdir(category_folder):
                model_folder = os.path.join(category_folder, object_id)
                if os.path.isdir(model_folder):
                    model_file = os.path.join(
                        model_folder, "models", "model_normalized.obj"
                    )
                    if os.path.isfile(model_file):
                        shapenet_data.append(
                            {
                                "object_id": object_id,
                                "category_id": category_id,
                                "category": category,
                            }
                        )

    shapenet_df = pd.DataFrame(shapenet_data)
    # shapenet_df.to_csv("shapenet_models.csv", index=False)

    return shapenet_df


def zero_shapenet_obj(mesh, target_size=100, return_transfrom=False, desired_xy=(0, 0)):
    # Rotate mesh to stand upright
    mesh = mesh.rotate_x(90)
    # Center mesh at origin
    mesh_width = mesh.bounds[1] - mesh.bounds[0]
    mesh_depth = mesh.bounds[3] - mesh.bounds[2]
    # scale to be 100 millimeters wide on the smallest side
    scale = target_size / max(mesh_width, mesh_depth)
    mesh = mesh.scale([scale, scale, scale])
    z_height = mesh.bounds[5] - mesh.bounds[4]
    translation = (
        np.array((0, 0, z_height / 2))
        - np.array(mesh.center)
        + np.array((desired_xy[0], desired_xy[1], 0))
    )
    mesh = mesh.translate(translation)

    if return_transfrom:
        return mesh, translation, scale
    else:
        return mesh


"""PartNet Helpers:"""


def tree_from_json(json_file):
    with open(json_file, "r") as f:
        json_obj = json.load(f)

    tree = treelib.Tree()

    def add_nodes(node, parent_id=None):
        node_id = node.get("id")
        node_name = node.get("name", "")
        node_objs = node.get("objs", [])

        # Use ori_id if id is not available
        if node_id is None:
            node_id = node.get("ori_id")

        if node_id is not None:
            # Use the objs array as the node data
            tree.create_node(
                tag=f"{node_name}",
                identifier=node_id,
                parent=parent_id,
                data=node_objs,
            )

            for child in node.get("children", []):
                add_nodes(child, parent_id=node_id)

    # Handle both single object and array input
    if isinstance(json_obj, list):
        for item in json_obj:
            add_nodes(item)
    else:
        add_nodes(json_obj)

    return tree


def get_partnet_sample(dir, max_parts=None, desired_xy=(0, 0)):
    sample = {}
    sample["meshes"] = get_and_transform_partnet_meshes(
        dir, max_parts=max_parts, desired_xy=desired_xy
    )
    if sample["meshes"] is None:
        return None
    sample["num_parts"] = len(sample["meshes"])
    with open(os.path.join(dir, "meta.json"), "r") as f:
        json_obj = json.load(f)
        sample["category"] = json_obj.get("model_cat", "")
        sample["model_id"] = json_obj.get("model_id", "")
    sample["part_tree"] = tree_from_json(os.path.join(dir, "result_after_merging.json"))
    return sample


def get_and_transform_partnet_meshes(dir, max_parts=None, desired_xy=(0, 0)):
    original_meshes = {}
    combined_mesh = pv.PolyData()

    files = [f for f in os.listdir(os.path.join(dir, "objs")) if f.endswith(".obj")]

    if max_parts is not None and len(files) > max_parts:
        return None

    for file in files:
        mesh_id = file.replace(".obj", "")
        mesh = pv.read(os.path.join(dir, "objs", file)).clean()
        original_meshes[mesh_id] = mesh
        combined_mesh = combined_mesh.merge(mesh)

    _, translation, scale = zero_shapenet_obj(
        combined_mesh, return_transfrom=True, desired_xy=desired_xy
    )

    transformed_meshes = {}
    for mesh_id, mesh in original_meshes.items():
        new_mesh = mesh.rotate_x(90)
        # scale to be 100 millimeters wide on the smallest side
        new_mesh = new_mesh.scale([scale, scale, scale])
        new_mesh = new_mesh.translate(translation)
        transformed_meshes[mesh_id] = new_mesh

    return transformed_meshes


def collect_partnet_data(
    dir="/Users/ryanslocum/Downloads/cutlist/PartNet-archive/data_v0",
):
    model_data = []
    for folder in os.listdir(dir):
        model_dir = os.path.join(dir, folder)
        part_count = len(os.listdir(os.path.join(model_dir, "objs")))
        meta_data = json.load(open(os.path.join(model_dir, "meta.json"), "r"))
        model_cat = meta_data.get("model_cat", "")
        model_id = meta_data.get("model_id", "")
        model_data.append(
            {
                "object_id": model_id,
                "category": model_cat,
                "model_dir": folder,
                "part_count": part_count,
            }
        )
    df = pd.DataFrame(model_data)
    # save to csv
    # df.to_csv(os.path.join(os.path.dirname(dir), "_model_data.csv"), index=False)
    return df


"""BrickGPT Helpers:"""


def get_brickgpt_data(dir="/Users/ryanslocum/Downloads/cutlist/StableText2Brick/data"):
    stabletext2brick_df = pd.DataFrame()
    for file in os.listdir(dir):
        if file.endswith(".parquet"):
            df = pd.read_parquet(os.path.join(dir, file))
            stabletext2brick_df = pd.concat(
                [stabletext2brick_df, df], ignore_index=True
            )
    # print column names
    # print(stabletext2brick_df.columns)
    return stabletext2brick_df


def merge_shapes_and_captions(objects_df, brickgpt_df):
    merged_df = pd.merge(
        objects_df, brickgpt_df[["object_id", "captions"]], on=["object_id"], how="left"
    )

    # remove duplicates
    merged_df = merged_df.drop_duplicates(subset=["object_id"])

    # Print how many models have captions and how many do not
    print(
        f"Models with captions: {merged_df['captions'].notnull().sum()}, Models without captions: {merged_df['captions'].isnull().sum()}"
    )
    # print the category of models that do not have captions
    print(merged_df[merged_df["captions"].isnull()]["category"].value_counts())

    # print ten random rows
    print(merged_df.sample(10))

    merged_df.to_csv("merged_csv.csv", index=False)

    return merged_df
