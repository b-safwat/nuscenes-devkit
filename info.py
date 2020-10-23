from nuscenes import NuScenes
from nuscenes.eval.prediction.splits import get_prediction_challenge_split
import numpy as np


def what_are_the_objects_types_used_in_prediction_tasks(DATAROOT='./data/sets/nuscenes', dataset_version='v1.0-mini'):
    nuscenes = NuScenes(dataset_version, DATAROOT)
    train_agents = get_prediction_challenge_split("train", dataroot=DATAROOT)
    val_agents = get_prediction_challenge_split("val", dataroot=DATAROOT)
    #
    # agents = get_prediction_challenge_split("train", dataroot=DATAROOT)
    agents = np.concatenate((train_agents, val_agents), axis=0)
    categories = {}
    token_to_name = {}

    for current_sample in agents:
        instance_token, sample_token = current_sample.split("_")
        category_token = nuscenes.get("instance", instance_token)["category_token"]
        category_name = nuscenes.get("category", category_token)["name"]
        categories[category_name] = category_token
        token_to_name[category_token] = category_name

    print(categories.items())
    print(token_to_name.items())

if __name__ == '__main__':
    # what_are_the_objects_types_used_in_prediction_tasks()
    what_are_the_objects_types_used_in_prediction_tasks('/media/bassel/Entertainment/nuscenes/v1.0-trainval01_blobs',
                                  'v1.0-trainval')


