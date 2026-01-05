import os
import json
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from src.transforms.functional import rgb_to_ycbcr


class FastVideoFolder(Dataset):
    def __init__(self, root_folder_path, patch_h, patch_w, frame_num, crop_method='center',
                 max_zoom_factor=1.0, min_zoom_factor=1.0, random_flip=False,
                 frame_selection='random', max_frame_distance=6, disable_random=False, Pyuv420=False):
        '''
        crop_method could be 'center', 'random'
        frame_selection could be 'random', 'fix'
        '''
        with open(os.path.join(root_folder_path, 'description.json')) as json_file:
            datasets = json.load(json_file)
        self.root_folder_path = root_folder_path
        self.seqs = []
        for dataset in datasets:
            dataset_name = dataset['dataset_name']
            duplicated = dataset['duplicated']
            duplicated = int(duplicated)
            duplicated = max(1, duplicated)
            dataset_desc_path = os.path.join(
                self.root_folder_path, dataset_name, 'description.json')
            with open(dataset_desc_path) as json_file:
                data = json.load(json_file)
                for _ in range(duplicated):
                    self.seqs.extend(data)

        self.dataset_length = len(self.seqs)
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.frame_num = frame_num
        self.crop_method = crop_method
        assert max_zoom_factor >= min_zoom_factor
        self.max_zoom_factor = max_zoom_factor
        self.min_zoom_factor = min_zoom_factor
        self.enable_random_zoom = max_zoom_factor > 1.01 or max_zoom_factor < 0.99 or \
            min_zoom_factor > 1.01 or min_zoom_factor < 0.99
        self.random_flip = random_flip
        self.frame_selection = frame_selection
        self.max_frame_distance = max_frame_distance
        self.disable_random = disable_random
        self.Pyuv420 = Pyuv420
        if self.disable_random:
            self.crop_method = 'center'
            self.max_zoom_factor = 1.0
            self.min_zoom_factor = 1.0
            self.enable_random_zoom = False
            self.random_flip = False
            self.frame_selection = 'fix'
            self.max_frame_distance = 1

    def set_frame_num(self, frame_num):
        self.frame_num = frame_num

    def get_frame_num(self):
        return self.frame_num

    def __getitem__(self, index):
        seq = self.seqs[index]
        height = seq['height']
        width = seq['width']
        seq_path = seq['path']

        img_indexes = []
        if self.frame_selection == 'fix':
            img_indexes = range(0, self.frame_num)
        elif self.frame_selection == 'random':
            if self.frame_num < seq["seq_length"]:
                img_indexes = random.sample(range(0, seq["seq_length"]), self.frame_num)
                is_reverse_order = random.choice([True, False])
                img_indexes.sort(reverse=is_reverse_order)
                for i in range(1, len(img_indexes), 1):
                    pre_index = img_indexes[i-1]
                    cur_index = img_indexes[i]
                    if is_reverse_order:
                        if cur_index < pre_index - self.max_frame_distance:
                            cur_index = random.randint(
                                pre_index - self.max_frame_distance, pre_index - 1)
                            img_indexes[i] = cur_index
                    else:
                        if cur_index > pre_index + self.max_frame_distance:
                            cur_index = random.randint(
                                pre_index + 1, pre_index + self.max_frame_distance)
                            img_indexes[i] = cur_index
            else:
                increasing = True
                frame_index = 0
                while len(img_indexes) < self.frame_num:
                    img_indexes.append(frame_index)
                    if increasing:
                        if frame_index == seq["seq_length"] - 1:
                            frame_index -= 1
                            increasing = False
                        else:
                            frame_index += 1
                    elif not increasing:
                        if frame_index == 0:
                            frame_index += 1
                            increasing = True
                        else:
                            frame_index -= 1
        else:
            assert False

        if self.enable_random_zoom:
            zoom_factor = random.uniform(self.min_zoom_factor, self.max_zoom_factor)
            scaled_width = int(width * zoom_factor)
            scaled_height = int(height * zoom_factor)
        else:
            scaled_width = width
            scaled_height = height
        flip = False
        if self.random_flip:
            flip = random.choice([True, False])

        pad_height = self.patch_h - scaled_height
        pad_width = self.patch_w - scaled_width
        pad_height = max(0, pad_height)
        pad_width = max(0, pad_width)
        pad_size = ((0, 0),
                    (pad_height // 2, pad_height - pad_height // 2),
                    (pad_width // 2, pad_width - pad_width // 2))
        padded_height = scaled_height + pad_height
        padded_width = scaled_width + pad_width
        if self.crop_method == 'center':
            y = (padded_height - self.patch_h) // 2
            x = (padded_width - self.patch_w) // 2
        elif self.crop_method == 'random':
            y = random.randint(0, padded_height - self.patch_h)
            x = random.randint(0, padded_width - self.patch_w)
        else:
            assert False

        video_data = []
        for img_index in img_indexes:
            img_path = os.path.join(self.root_folder_path, seq_path, seq['frames'][img_index])
            img = Image.open(img_path).convert("RGB")
            if self.enable_random_zoom:
                img = img.resize((scaled_width, scaled_height), Image.BILINEAR)
            if flip:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            img = np.array(img).transpose(2, 0, 1).astype(np.uint8)
            img = np.pad(img, pad_size, mode='constant')

            img = img[:, y:y+self.patch_h, x:x+self.patch_w]
            #######################
            img = img.astype(np.float32) / 255.0
            #rgb to yuv444
            if self.Pyuv420:
                img = rgb_to_ycbcr(img)
            #######################
            video_data.append(img)

        video_data = np.concatenate(video_data, axis=0)
        return torch.as_tensor(video_data.astype(np.float32), dtype=torch.float32)

    def __len__(self):
        return self.dataset_length
