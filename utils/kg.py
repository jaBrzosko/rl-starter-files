import torch
import gymnasium as gym
import torch_ac
import numpy as np

def test_reshape(observation, action, reward, done):
    # print("Original reward:", reward)
    # print("Action taken:", action)
    # print("Observation:", observation)
    # print("Done:", done)

    return reward

def test_preobs(observation, device):
    if isinstance(observation, list):
        obss = [process_single_environment_observation(obs) for obs in observation]
    elif isinstance(observation, tuple):
        obss = [process_single_environment_observation(obs) for obs in observation]
    else:
        obss = [process_single_environment_observation(observation)]

    return torch_ac.DictList({
        "image": torch.tensor(np.array(obss), device=device, dtype=torch.float)
    })

def process_single_environment_observation(observation):
    if isinstance(observation, tuple):
        observation = observation[0]

    if isinstance(observation, gym.spaces.Box):
        image = observation
    elif isinstance(observation, gym.spaces.Dict) and "image" in observation.spaces.keys():
        image = observation.spaces["image"]
    elif isinstance(observation, dict) and "image" in observation.keys():
        image = observation["image"]
    else:
        raise ValueError("Unknown observation space: " + str(observation))

    processed_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.float32)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            processed_image[i, j, 0] = process_single_tile(image[i, j])

    return processed_image

def process_single_tile(tile):
    assert len(tile) == 3, "Tile must have 3 channels"
    object_id = tile[0]
    # ignore color
    state = tile[2]

    if object_id == 0:  # unseen
        return 0
    elif object_id == 1:  # empty
        return 1
    elif object_id == 2:  # wall
        return 2
    elif object_id == 4: # door
        if state == 0: # open
            return 3
        elif state == 1: # closed
            return 4
        elif state == 2: # locked
            return 5
    elif object_id == 5: # key
        return 6
    elif object_id == 8: # goal
        return 7
    else:
        raise ValueError("Unknown object id: " + str(object_id))
