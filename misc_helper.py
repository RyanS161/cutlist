from nltk.corpus import wordnet as wn
import os
import pandas as pd
import json


def offset_to_word(offset, pos="n"):
    # offset: string or int, e.g. '02691156'
    # pos: part of speech, 'n' for noun (most ShapeNet classes are nouns)
    synset = wn.synset_from_pos_and_offset(pos, int(offset))
    return synset.name(), synset.definition()


def get_shapenet_words():
    for file in os.listdir("ShapeNetCore"):
        offset = file.replace(".zip", "")
        if offset.isdigit():
            name, definition = offset_to_word(offset)
            print(f"{offset}: {name} - {definition}")


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


partnet_df = collect_partnet_data(
    dir="/Users/ryanslocum/Downloads/cutlist/PartNet-archive/data_v0"
)
brick_gpt_df = get_brickgpt_data(
    dir="/Users/ryanslocum/Downloads/cutlist/StableText2Brick/data"
)

merge_shapes_and_captions(partnet_df, brick_gpt_df)
