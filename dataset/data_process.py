import os
import shutil
import torch
import tensorflow_datasets as tfds
import json
import numpy as np
import imageio
import random
import multiprocessing as mp
import gc
from tqdm import tqdm
from torchvision import transforms


# Hide those pesky warnings.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def _load_episode_droid(episode: dict):

    step_index = 0
    actions = []
    cartesian_positions = []
    gripper_positions = []
    images_0 = []
    images_1 = []
    images_2 = []

    save_path = episode['save_path']
    split = episode['split']
    episode_index = str(episode['index'])
    episode_dict = dict(episode_id=episode_index)

    episode_video_path = os.path.join(save_path, 'videos', split, episode_index)
    split_anno_path = os.path.join(save_path, 'annotation', split)
    os.makedirs(episode_video_path, exist_ok=True)
    os.makedirs(split_anno_path, exist_ok=True)
    episode_anno_path = os.path.join(split_anno_path, f'{str(episode_index)}.json')

    transform = transforms.CenterCrop((176, 320))

    for step in episode['data']:

        if step_index == 0:
            # load prompts
            episode_dict.update(
                language_instruction=[
                    bytearray(step["language_instruction"].numpy()).decode(),
                    # bytearray(step["language_instruction_2"].numpy()).decode(),
                    # bytearray(step["language_instruction_3"].numpy()).decode(),
                ],
            )

        # load images and actions
        observation = step['observation']
        cartesian_positions.append(observation['cartesian_position'].numpy())
        gripper_positions.append(observation['gripper_position'].numpy()[0])
        images_0.append(observation['exterior_image_1_left'].numpy())  # [180, 320, 3]
        images_1.append(observation['exterior_image_2_left'].numpy())
        images_2.append(observation['wrist_image_left'].numpy())

        # load states
        action_dict = step['action_dict']
        actions.append(
            np.concatenate([
                action_dict['cartesian_velocity'].numpy(),
                action_dict['gripper_position'].numpy(),
            ])
        )

    # post process
    actions = np.array(actions)
    cartesian_positions = np.array(cartesian_positions)
    gripper_positions = np.array(gripper_positions)

    images_0 = np.array(images_0)
    images_1 = np.array(images_1)
    images_2 = np.array(images_2)
    images_0 = transform(
        torch.from_numpy(images_0).permute(0, 3, 1, 2)
    ).permute(0, 2, 3, 1).numpy()
    images_1 = transform(
        torch.from_numpy(images_1).permute(0, 3, 1, 2)
    ).permute(0, 2, 3, 1).numpy()
    images_2 = transform(
        torch.from_numpy(images_2).permute(0, 3, 1, 2)
    ).permute(0, 2, 3, 1).numpy()

    try:

        # save videos
        video_params = dict(
            fps=10,
            codec='libx264',
            macro_block_size=None,
            ffmpeg_params=["-crf", "20", "-preset", "veryfast"],
        )
        imageio.mimwrite(
            video_path_0 := os.path.join(
                episode_video_path,
                'exterior_image_1_left.mp4'
            ),
            images_0,
            **video_params,
        )
        imageio.mimwrite(
            video_path_1 := os.path.join(
                episode_video_path,
                'exterior_image_2_left.mp4'
            ),
            images_1,
            **video_params,
        )
        imageio.mimwrite(
            video_path_2 := os.path.join(
                episode_video_path,
                'wrist_image_left.mp4'
            ),
            images_2,
            **video_params,
        )
        episode_dict.update(
            videos=[
                {'video_path': os.path.relpath(video_path_0, save_path)},
                {'video_path': os.path.relpath(video_path_1, save_path)},
                {'video_path': os.path.relpath(video_path_2, save_path)}
            ]
        )

        # save states
        episode_dict.update(
            action=actions.tolist(),
            state=cartesian_positions.tolist(),
            continuous_gripper_state=gripper_positions.tolist(),
        )

        # save annotation
        with open(episode_anno_path, 'w', encoding='utf-8') as f:
            json.dump(episode_dict, f, ensure_ascii=False, indent=4)

    except Exception as e:

        if os.path.exists(episode_video_path):
            shutil.rmtree(episode_video_path)
        if os.path.exists(episode_anno_path):
            os.remove(episode_anno_path)


def _load_episode_bridgev2(episode: dict):

    step_index = 0
    actions = []
    states = []
    images_0 = []
    images_1 = []
    images_2 = []
    images_3 = []
    continuous_gripper_state = []

    has_image_0 = bool(episode['episode_metadata']['has_image_0'])
    has_image_1 = bool(episode['episode_metadata']['has_image_1'])
    has_image_2 = bool(episode['episode_metadata']['has_image_2'])
    has_image_3 = bool(episode['episode_metadata']['has_image_3'])

    # check if has_image_0/1/2/3
    observation = episode['data'][0]['observation']
    image_0 = observation['image_0'].numpy()
    image_1 = observation['image_1'].numpy()
    image_2 = observation['image_2'].numpy()
    image_3 = observation['image_3'].numpy()
    has_image_0 = bool(np.sum(image_0) > 0.)
    has_image_1 = bool(np.sum(image_1) > 0.)
    has_image_2 = bool(np.sum(image_2) > 0.)
    has_image_3 = bool(np.sum(image_3) > 0.)

    save_path = episode['save_path']
    split = episode['split']
    episode_index = str(episode['index'])
    episode_dict = dict(
                episode_id=episode_index,
                has_image_0=has_image_0,
                has_image_1=has_image_1,
                has_image_2=has_image_2,
                has_image_3=has_image_3,
    )

    episode_video_path = os.path.join(save_path, 'videos', split, episode_index)
    split_anno_path = os.path.join(save_path, 'annotation', split)
    os.makedirs(episode_video_path, exist_ok=True)
    os.makedirs(split_anno_path, exist_ok=True)
    episode_anno_path = os.path.join(split_anno_path, f'{str(episode_index)}.json')

    transform = transforms.Resize((480, 640))

    for step in episode['data']:

        if step_index == 0:
            # load prompts
            episode_dict.update(
                texts=[
                    bytearray(step['language_instruction'].numpy()).decode()
                    if bool(episode['episode_metadata']['has_language']) else '',
                ],
            )

        # load images and actions
        observation = step['observation']
        if has_image_0:
            images_0.append(observation['image_0'].numpy())  # [256, 320, 3]
        if has_image_1:
            images_1.append(observation['image_1'].numpy())
        if has_image_2:
            images_2.append(observation['image_2'].numpy())
        if has_image_3:
            images_3.append(observation['image_3'].numpy())

        # load states
        actions.append(step['action'])
        states.append(observation['state'])
        continuous_gripper_state.append(observation['state'][6])

    # post process
    actions = np.array(actions)
    states = np.array(states)
    continuous_gripper_state = np.array(continuous_gripper_state)

    if images_0:
        images_0 = np.array(images_0)
        images_0 = transform(
            torch.from_numpy(images_0).permute(0, 3, 1, 2)
        ).permute(0, 2, 3, 1).numpy()
    if images_1:
        images_1 = np.array(images_1)
        images_1 = transform(
            torch.from_numpy(images_1).permute(0, 3, 1, 2)
        ).permute(0, 2, 3, 1).numpy()
    if images_2:
        images_2 = np.array(images_2)
        images_2 = transform(
            torch.from_numpy(images_2).permute(0, 3, 1, 2)
        ).permute(0, 2, 3, 1).numpy()
    if images_3:
        images_3 = np.array(images_3)
        images_3 = transform(
            torch.from_numpy(images_3).permute(0, 3, 1, 2)
        ).permute(0, 2, 3, 1).numpy()

    try:

        # save videos
        video_params = dict(
            fps=10,
            codec='libx264',
            macro_block_size=None,
            ffmpeg_params=["-crf", "20", "-preset", "veryfast"],
        )
        if has_image_0:
            imageio.mimwrite(
                video_path_0 := os.path.join(
                    episode_video_path,
                    'image_0.mp4'
                ),
                images_0,
                **video_params,
            )
        if has_image_1:
            imageio.mimwrite(
                video_path_1 := os.path.join(
                    episode_video_path,
                    'image_1.mp4'
                ),
                images_1,
                **video_params,
            )
        if has_image_2:
            imageio.mimwrite(
                video_path_2 := os.path.join(
                    episode_video_path,
                    'image_2.mp4'
                ),
                images_2,
                **video_params,
            )
        if has_image_3:
            imageio.mimwrite(
                video_path_3 := os.path.join(
                    episode_video_path,
                    'image_3.mp4'
                ),
                images_3,
                **video_params,
            )
        episode_dict.update(
            videos=[
                {'video_path': os.path.relpath(video_path_0, save_path)
                                if has_image_0 else ''},
                {'video_path': os.path.relpath(video_path_1, save_path)
                                if has_image_1 else ''},
                {'video_path': os.path.relpath(video_path_2, save_path)
                                if has_image_2 else ''},
                {'video_path': os.path.relpath(video_path_3, save_path)
                                if has_image_3 else ''},
            ]
        )

        # save states
        episode_dict.update(
            action=actions.tolist(),
            state=states.tolist(),
            continuous_gripper_state=continuous_gripper_state.tolist(),
        )

        # save annotation
        with open(episode_anno_path, 'w', encoding='utf-8') as f:
            json.dump(episode_dict, f, ensure_ascii=False, indent=4)

    except Exception as e:

        if os.path.exists(episode_video_path):
            shutil.rmtree(episode_video_path)
        if os.path.exists(episode_anno_path):
            os.remove(episode_anno_path)


def episode_loader(queue, load_func):
    pid = mp.current_process().name
    tqdm.write(f'Starting episode loader process {pid}')

    while True:
        episode = queue.get()
        if episode is None:
            tqdm.write(f'Ending episode loader process {pid}')
            break
        else:
            tqdm.write(f'Episode loader {pid} starts a new process')
            try:
                load_func(episode)
                del episode
            except Exception as e:
                tqdm.write(f"Skipped {episode['index']} due to {e}!")


def process_droid():

    num_workers = 8

    def _preprocess_episode(episode, index, split, save_path):
        preprocessed_episode = dict(index=index, split=split, data=[], save_path=save_path)
        for step in episode['steps']:
            preprocessed_episode['data'].append(
                {
                    'language_instruction': step['language_instruction'],
                    'observation': step['observation'],
                    'action_dict': step['action_dict'],
                }
            )
        del episode
        return preprocessed_episode

    # total 1192-of-2048 tfds data and 53586 episodes
    data_path = './data/droid_tfds'
    save_path = './data/droid'

    # load tensorflow dataset
    ds = tfds.builder_from_directory(builder_dir=data_path).as_dataset()
    train_ds = ds['train']
    print(f'Loaded dataset from {data_path} ...')
    train_ds = train_ds.prefetch(2)

    # split trainval!!!
    total_episodes = 29537
    train_episodes = int(total_episodes * 0.95)
    val_episodes = int(total_episodes - train_episodes)

    train_episode_index = random.sample(range(total_episodes), train_episodes)
    val_episode_index = list(set(range(total_episodes)) - set(train_episode_index))

    # multiprocessing setup
    context = mp.get_context('spawn')
    queue = context.Queue(maxsize=num_workers)

    # start episode loader process
    episode_loader_processes = []
    for i in range(num_workers):
        p = context.Process(
            target=episode_loader,
            args=(queue, _load_episode_droid),
            name=f'EpisodeLoaderProcess-{i}',
        )
        p.start()
        episode_loader_processes.append(p)

    for index, episode in enumerate(tqdm(train_ds, total=total_episodes)):
        split = 'train' if index in train_episode_index else 'val'
        # if already processed, just skip.
        if (
            os.path.exists(os.path.join(save_path, 'videos', split, str(index))) and
            os.path.exists(os.path.join(save_path, 'annotation', split, f'{str(index)}.json'))
        ):
            continue
        if index >= total_episodes:
            break
        tqdm.write(f'Processing episode {index=} for {split=}')
        preprocessed_episode = _preprocess_episode(episode, index, split, save_path=save_path)
        queue.put(preprocessed_episode, block=True, timeout=None)
        gc.collect()

    # send termination signals to loaders
    for _ in range(num_workers):
        queue.put(None)

    # wait for all processes to complete
    for p in episode_loader_processes:
        p.join()


def process_bridge():

    num_workers = 16

    def _preprocess_episode(episode, index, split, save_path):
        preprocessed_episode = dict(index=index, split=split, data=[], save_path=save_path)
        for step in episode['steps']:
            preprocessed_episode['data'].append(
                {
                    'language_instruction': step['language_instruction'],
                    'observation': step['observation'],
                    'action': step['action'],
                }
            )
        preprocessed_episode['episode_metadata'] = episode['episode_metadata']
        del episode
        return preprocessed_episode

    data_path = './data/bridgev2_tfds'
    save_path = './data/bridgev2'

    # load tensorflow dataset
    ds = tfds.builder_from_directory(builder_dir=data_path).as_dataset()
    train_ds = ds['train']
    val_ds = ds['val']
    print(f'Loaded dataset from {data_path} ...')
    train_ds = train_ds.prefetch(2)
    val_ds = val_ds.prefetch(2)

    # split trainval!!!
    train_episodes_limit = int(25460)  # 10K episodes
    val_episodes_limit = int(1737)  # 1K episodes

    # multiprocessing setup
    context = mp.get_context('spawn')
    queue = context.Queue(maxsize=num_workers)

    # start episode loader process
    episode_loader_processes = []
    for i in range(num_workers):
        p = context.Process(
            target=episode_loader,
            args=(queue, _load_episode_bridgev2),
            name=f'EpisodeLoaderProcess-{i}',
        )
        p.start()
        episode_loader_processes.append(p)

    # process train split
    # for index, episode in enumerate(tqdm(train_ds, total=train_episodes_limit)):
    #     # if already processed, just skip.
    #     if (
    #         os.path.exists(os.path.join(save_path, 'videos', 'train', str(index))) and
    #         os.path.exists(os.path.join(save_path, 'annotation', 'train', f'{str(index)}.json'))
    #     ):
    #         continue
    #     if index >= train_episodes_limit:
    #         break
    #     tqdm.write(f'Processing episode {index=} for train split ...')
    #     preprocessed_episode = _preprocess_episode(episode, index, 'train', save_path=save_path)
    #     queue.put(preprocessed_episode, block=True, timeout=None)
    #     gc.collect()

    # process val split
    for index, episode in enumerate(tqdm(val_ds, total=val_episodes_limit)):
        # if already processed, just skip.
        if (
            os.path.exists(os.path.join(save_path, 'videos', 'val', str(index))) and
            os.path.exists(os.path.join(save_path, 'annotation', 'val', f'{str(index)}.json'))
        ):
            continue
        if index >= val_episodes_limit:
            break
        tqdm.write(f'Processing episode {index=} for val split ...')
        preprocessed_episode = _preprocess_episode(episode, index, 'val', save_path=save_path)
        queue.put(preprocessed_episode, block=True, timeout=None)
        gc.collect()

    # send termination signals to loaders
    for _ in range(num_workers):
        queue.put(None)

    # wait for all processes to complete
    for p in episode_loader_processes:
        p.join()


if __name__ == "__main__":
    # process_droid()
    process_bridge()
