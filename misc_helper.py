from nltk.corpus import wordnet as wn
import os
import pandas as pd

def offset_to_word(offset, pos='n'):
    # offset: string or int, e.g. '02691156'
    # pos: part of speech, 'n' for noun (most ShapeNet classes are nouns)
    synset = wn.synset_from_pos_and_offset(pos, int(offset))
    return synset.name(), synset.definition()

def get_shapenet_words():

    for file in os.listdir('ShapeNetCore'):
        offset = file.replace('.zip', '')
        if offset.isdigit():
            name, definition = offset_to_word(offset)
            print(f"{offset}: {name} - {definition}")


def collect_shapenet_data():

    shapenet_data = []
    shapenet_parent_dir = "/Users/ryanslocum/Downloads/GenWood/ShapeNetCore"
    for category_id in os.listdir(shapenet_parent_dir):
        category_folder = os.path.join(shapenet_parent_dir, category_id)
        if os.path.isdir(category_folder):
            if not category_id.isdigit():
                continue
            category = offset_to_word(category_id)[0]
            print(f"Entering Category: {category}, Offset: {category_id}")
            for object_id in os.listdir(category_folder):
                model_folder = os.path.join(category_folder, object_id)
                if os.path.isdir(model_folder):
                    model_file = os.path.join(model_folder, "models", "model_normalized.obj")
                    if os.path.isfile(model_file):
                        shapenet_data.append((category_id, category, object_id))

    shapenet_df = pd.DataFrame(shapenet_data, columns=["category_id", "category", "object_id"])
    # Print how many models per category
    print(shapenet_df['category'].value_counts())
    shapenet_df.to_csv("shapenet_models.csv", index=False)

    stabletext2brick_parent_dir = "/Users/ryanslocum/Downloads/GenWood/StableText2Brick/data"
    stabletext2brick_df = pd.DataFrame()
    for file in os.listdir(stabletext2brick_parent_dir):
        if file.endswith(".parquet"):
            df = pd.read_parquet(os.path.join(stabletext2brick_parent_dir, file))
            stabletext2brick_df = pd.concat([stabletext2brick_df, df], ignore_index=True)
    # print column names
    print(stabletext2brick_df.columns)


    # Add "captions" column to shapenet_df on matching category_id and object_id
    merged_df = pd.merge(shapenet_df, stabletext2brick_df[['category_id', 'object_id', 'captions']], on=['category_id', 'object_id'], how='left')

    # remove duplicates
    merged_df = merged_df.drop_duplicates(subset=['category_id', 'object_id'])

    # Print how many models have captions and how many do not
    print(merged_df['captions'].notnull().sum(), merged_df['captions'].isnull().sum())
    # print the category of models that do not have captions
    print(merged_df[merged_df['captions'].isnull()]['category'].value_counts())


    merged_df.to_csv(os.path.join(shapenet_parent_dir, "shapenet_models_with_captions.csv"), index=False)

collect_shapenet_data()