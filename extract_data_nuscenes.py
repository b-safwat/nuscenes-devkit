import math
import numpy as np

from nuscenes import NuScenes
from nuscenes.eval.common.utils import quaternion_yaw
from nuscenes.eval.prediction.splits import get_prediction_challenge_split
from nuscenes.prediction import PredictHelper
from nuscenes.prediction.helper import angle_of_rotation
from pyquaternion import Quaternion

from Formats import FORMAT_FOR_MODEL


class NuScenesFormatTransformer:
    def __init__(self, DATAROOT='./data/sets/nuscenes', dataset_version='v1.0-mini'):
        self.DATAROOT = DATAROOT
        self.nuscenes = NuScenes(dataset_version, dataroot=self.DATAROOT)
        self.helper = PredictHelper(self.nuscenes)
        # ['vehicle.car', 'vehicle.truck', 'vehicle.bus.rigid', 'vehicle.bus.bendy', 'vehicle.construction']
        self.category_token_to_id = {"fd69059b62a3469fbaef25340c0eab7f":1, # 'vehicle.car'
                                    "6021b5187b924d64be64a702e5570edf":1,  # 'vehicle.truck'
                                    "fedb11688db84088883945752e480c2c":2,  # 'vehicle.bus.rigid'
                                    "003edbfb9ca849ee8a7496e9af3025d4":2,  # 'vehicle.bus.bendy'
                                    "5b3cd6f2bca64b83aa3d0008df87d0e4":3,  # 'vehicle.construction'
                                     "7b2ff083a64e4d53809ae5d9be563504":1} # vehicle.emergency.police

    def get_new_format(self, samples_agents, format_for_model, out_file=("./transformer_format.txt"), num_seconds=None):
        # for current_sample in samples_agents:
        #     instance_token, sample_token = current_sample.split("_")
        #     traj = self.helper.get_future_for_agent(instance_token, sample_token, 6, True)
        #     past_traj = self.helper.get_past_for_agent(instance_token, sample_token, 6, True)
        #
        #     if len(past_traj) + len(traj) + 1 < 20:
        #         print(len(past_traj) + len(traj) + 1)
        #
        # exit()
        ####################
        # Sample Token (frame) to a sequential id
        # for each sample (agent_frame), get the scene it belongs to, and then get the first sample (frame)
        # loop on all samples from the first sample till the end
        # set to dictionary the sequential id for each sample
        splitting_format = '\t'

        if format_for_model.value == FORMAT_FOR_MODEL.TRAFFIC_PREDICT.value:
            splitting_format = " "
            
        instance_token_to_id_dict = {}
        sample_token_to_id_dict = {}
        scene_token_dict = {}
        sample_id = 0
        instance_id = 0

        for current_sample in samples_agents:
            instance_token, sample_token = current_sample.split("_")
            scene_token = self.nuscenes.get('sample', sample_token)["scene_token"]

            if scene_token in scene_token_dict:
                continue

            # get the first sample in this sequence
            scene_token_dict[scene_token] = True
            first_sample_token = self.nuscenes.get("scene", scene_token)["first_sample_token"]
            current_sample = self.nuscenes.get('sample', first_sample_token)

            while True:
                if current_sample['token'] not in sample_token_to_id_dict:
                    sample_token_to_id_dict[current_sample['token']] = sample_id
                    sample_id += 1
                else:
                    print("should not happen?")

                instances_in_sample = self.helper.get_annotations_for_sample(current_sample['token'])

                for sample_instance in instances_in_sample:
                    if sample_instance['instance_token'] not in instance_token_to_id_dict:
                        instance_token_to_id_dict[sample_instance['instance_token']] = instance_id
                        instance_id += 1

                if current_sample['next'] == "":
                    break

                current_sample = self.nuscenes.get('sample', current_sample['next'])

        #############
        # Converting to the transformer network format
        # frame_id, agent_id, pos_x, pos_y
        # todo:
        # loop on all the agents, if agent not taken:
        # 1- add it to takens agents (do not retake the agent)
        # 2- get the number of appearance of this agent
        # 3- skip this agent if the number is less than 10s (4 + 6)
        # 4- get the middle agent's token
        # 5- get the past and future agent's locations relative to its location
        samples_new_format = []
        taken_instances = {}
        ds_size = 0

        for current_sample in samples_agents:
            instance_token, sample_token = current_sample.split("_")
            instance_id, sample_id = instance_token_to_id_dict[instance_token], sample_token_to_id_dict[sample_token]

            if instance_id in taken_instances:
                continue

            taken_instances[instance_id] = True

            trajectory = self.get_trajectory_around_sample(instance_token, sample_token)
            trajectory_full_instances =  self.get_trajectory_around_sample(instance_token, sample_token,just_xy=False)
            # traj_samples_token = [instance['sample_token'] for instance in trajectory_full_instances]

            if len(trajectory) < 20:
                print("length is less than 20 samples, trajectory length is: ", len(trajectory))
                continue

            ds_size += 1

            if num_seconds is not None:
                start, end = len(trajectory)//2-9, len(trajectory)//2+11,
                starting_frame = (start+end)//2

                middle_sample_token = trajectory_full_instances[starting_frame]["sample_token"]
                trajectory = self.get_trajectory_around_sample(instance_token, middle_sample_token,
                                                               just_xy=True, num_seconds=num_seconds,
                                                               in_agent_frame=True)
                trajectory_full_instances = self.get_trajectory_around_sample(instance_token, middle_sample_token,
                                                                              just_xy=False, num_seconds=num_seconds,
                                                                              in_agent_frame=True)
                # traj_samples_token = [instance['sample_token'] for instance in trajectory_full_instances]

            # get_trajectory at this position
            for i in range(trajectory.shape[0]):
                traj_sample, sample_token = trajectory[i], trajectory_full_instances[i]["sample_token"]
                sample_id = sample_token_to_id_dict[sample_token]
                if format_for_model.value == FORMAT_FOR_MODEL.TRANSFORMER_NET.value:
                    samples_new_format.append(str(sample_id) + splitting_format + str(instance_id)\
                                           + splitting_format + str(traj_sample[0]) + splitting_format + str(traj_sample[1]) + "\n")
                elif format_for_model.value == FORMAT_FOR_MODEL.TRAFFIC_PREDICT.value:
                    # raise Exception("not implemented yet")
                    category_token = self.nuscenes.get("instance", instance_token)["category_token"]
                    object_type = self.category_token_to_id[category_token]
                    # frame_id, object_id, object_type,
                    # position_x, position_y, position_z,
                    # object_length, object_width, object_height,
                    # heading
                    x, y, z = trajectory_full_instances[i]["translation"]
                    w,l,h = trajectory_full_instances[i]["size"]
                    # yaw = angle_of_rotation(quaternion_yaw(Quaternion(trajectory_full_instances[i]["rotation"])))
                    yaw = quaternion_yaw(Quaternion(trajectory_full_instances[i]["rotation"]))

                    samples_new_format.append(str(sample_id) + splitting_format + str(instance_id) + splitting_format + str(object_type)\
                                              + splitting_format + str(x) + splitting_format + str(y) + splitting_format + str(z) + splitting_format
                                              + splitting_format + str(l) + splitting_format + str(w) + splitting_format + str(h) + splitting_format
                                              + str(yaw) + "\n")
            # annotations = helper.get_annotations_for_sample(sample_token)

            # for ann in annotations:
            #     # q = ann['rotation']
            #     # yaw = math.atan2(2.0 * (q[3] * q[0] + q[1] * q[2]), - 1.0 + 2.0 * (q[0] * q[0] + q[1] * q[1]))*180/math.pi
            #     # if yaw < 0:
            #     #     yaw += 360
            #     # selected_sample_data = [sample_id, instance_id] + ann['translation'] + [yaw] + ann['size']
            #     selected_sample_data = str(sample_id) + " " + str(instance_token_to_id_dict[ann['instance_token']])\
            #                            + " " + str(ann['translation'][0]) + " " + str(ann['translation'][2]) + "\n"
            #     samples_new_format.append(selected_sample_data)

        # no need for sorting as it occurs in the TransformationNet data loader
        # left it for similarity

        samples_new_format.sort(key=lambda x: int(x.split(splitting_format)[0]))

        with open(out_file, 'w') as fw:
            fw.writelines(samples_new_format)

        print(out_file + "size " + str(ds_size))

    def run(self, format_for_model):
        # train_agents = get_prediction_challenge_split("mini_train", dataroot=self.DATAROOT)
        # val_agents = get_prediction_challenge_split("mini_val", dataroot=self.DATAROOT)
        train_agents = get_prediction_challenge_split("train", dataroot=self.DATAROOT)
        val_agents = get_prediction_challenge_split("val", dataroot=self.DATAROOT)
        self.get_new_format(train_agents, format_for_model, "transformer_train.txt")
        self.get_new_format(val_agents, format_for_model, "transformer_val.txt")

    def get_trajectory_around_sample(self, instance_token, sample_token, just_xy=True,
                                     num_seconds=1000, in_agent_frame=False):
        future_samples = self.helper.get_future_for_agent(instance_token, sample_token,
                                                          num_seconds, in_agent_frame, just_xy)
        past_samples = self.helper.get_past_for_agent(instance_token, sample_token,
                                                      num_seconds, in_agent_frame, just_xy)[::-1]
        current_sample = self.helper.get_sample_annotation(instance_token, sample_token)

        if num_seconds == 5:
            if len(past_samples) > 9:
                past_samples = past_samples[0:9]
            if len(future_samples) > 10:
                future_samples = future_samples[0:10]

        if just_xy:
            current_sample = current_sample["translation"][:2]
            trajectory = np.append(past_samples, np.append([current_sample], future_samples, axis=0), axis=0)
        else:
            trajectory = np.append(past_samples, np.append([current_sample], future_samples, axis=0), axis=0)
        return trajectory


if __name__ == '__main__':
    n = NuScenesFormatTransformer('/media/bassel/Entertainment/nuscenes/v1.0-trainval01_blobs',
                                  'v1.0-trainval')
    # n = NuScenesFormatTransformer()

    # n.run( FORMAT_FOR_MODEL.TRANSFORMER_NET)
    n.run(FORMAT_FOR_MODEL.TRAFFIC_PREDICT)