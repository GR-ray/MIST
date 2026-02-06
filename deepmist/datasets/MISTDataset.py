import os
import torch
import torch.utils.data as Data
import torchvision.transforms as transforms
from glob import glob
from deepmist.utils import (rgb_loader, binary_loader, random_flip, random_crop, random_rotation, color_enhance,
                            random_peper)


class MISTDataset(Data.Dataset):
    def __init__(self, root, num_inputs=5, img_size=None, frame_padding=False, data_aug=None, split='train'):
        split_file = os.path.join(root, f'{split}.txt')
        if not os.path.isfile(split_file):
            raise ValueError(
                f"Invalid split '{split}'. It must be 'train', 'val_all' or 'val_hard'.")
        with open(split_file, 'r') as f:
            seq_list = [line.strip() for line in f if line.strip()]

        if img_size is None:
            img_size = [384, 384]
        self.img_size = img_size

        if data_aug is None:
            data_aug = {
                'random_flip': True,
                'random_crop': True,
                'random_rotation': True,
                'color_enhance': False,
                'random_peper': False
            }
        self.num_inputs = num_inputs
        self.data_aug = data_aug
        self.split = split
        self.grouped_frame_paths = []
        self.grouped_mask_paths = []

        # grouping
        for seq in seq_list:
            frame_paths = sorted(glob(os.path.join(root, 'image', seq, '*.png')))
            mask_paths = sorted(glob(os.path.join(root, 'mask', seq, '*.png')))
            num_frames = len(frame_paths)
            assert num_inputs <= num_frames, f"number of input frames '{num_inputs}' exceeds the total number."
            if frame_padding:
                for i in range(num_frames):
                    frame_list = []
                    for j in range(num_inputs - 1, -1, -1):
                        if i - j < 0:
                            frame_list.append(frame_paths[0])
                        else:
                            frame_list.append(frame_paths[i - j])
                    self.grouped_frame_paths.append(frame_list)
                    self.grouped_mask_paths.append(mask_paths[i])
            else:
                for i in range(num_inputs - 1, num_frames):
                    frame_list = []
                    for j in range(num_inputs - 1, -1, -1):
                        frame_list.append(frame_paths[i - j])
                    self.grouped_frame_paths.append(frame_list)
                    self.grouped_mask_paths.append(mask_paths[i])

        # transforms
        self.frame_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.359, 0.359, 0.359], [0.15, 0.15, 0.15])])
        self.mask_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor()])

    def __getitem__(self, index):
        frames = []
        for i in range(self.num_inputs):
            frames.append(rgb_loader(self.grouped_frame_paths[index][i]))
        mask = binary_loader(self.grouped_mask_paths[index])

        # data augmentation (only for training)
        if self.split == 'train':
            if self.data_aug['random_flip']:
                frames, mask = random_flip(frames, mask)
            if self.data_aug['random_crop']:
                frames, mask = random_crop(frames, mask)
            if self.data_aug['random_rotation']:
                frames, mask = random_rotation(frames, mask)
            if self.data_aug['color_enhance']:
                frames = color_enhance(frames)
            if self.data_aug['random_peper']:
                mask = random_peper(mask)

        # transforms
        for i in range(len(frames)):
            frames[i] = self.frame_transform(frames[i])
        frames = torch.stack(frames, dim=1)
        mask = self.mask_transform(mask)

        # get name
        mask_path_split = self.grouped_mask_paths[index].split('/')
        name = mask_path_split[-2] + '/' + mask_path_split[-1]

        return frames, mask, self.img_size[0], self.img_size[1], name

    def __len__(self):
        return len(self.grouped_frame_paths)
