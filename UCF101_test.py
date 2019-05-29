from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import random
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

D = 32

class ClipSubstractMean(object):
    def __init__(self, b=104, g=117, r=123):
        self.means = np.array((r,g,b))

    def __call__(self, sample):
        video_x, video_label = sample['video_x'], sample['video_label']
        new_video_x = video_x - self.means
        return {'video_x':new_video_x, 'video_label':video_label}

class Rescale(object):
    """
    Rescale the image in a sample to a given size.
    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
                                    matched to output_size. If int, smaller 
                                    of image edges is matched to output_size 
                                    keeping aspect ratio the same.
    
    """

    def __init__(self, output_size = (112,112)):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        video_x, video_label = sample['video_x'], sample['video_label']

        depth = D
        h, w = video_x.shape[1], video_x.shape[2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        new_video_x = np.zeros((depth, new_h, new_w, 3))
        for i in range(depth):
            image = video_x[i,:,:,:]
            image = transform.resize(image, (new_h, new_w))
            new_video_x[i,:,:,:] = image
        
        return {'video_x': new_video_x, 'video_label': video_label}

class RandomCrop(object):
    """
    Crop randomly the image in a sample.
    Args:
        output_size (tuple or int): Desired output size. If int, square crop
                                    is made.

    """

    def __init__(self, output_size = (112,112)):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        video_x, video_label = sample['video_x'], sample['video_label']
        
        depth = D
        h, w = video_x.shape[1], video_x.shape[2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        new_video_x = np.zeros((depth, new_h, new_w, 3))
        for i in range(depth):
            image = video_x[i,:,:,:]
            image = image[top : top + new_h, left : left + new_w]
            new_video_x[i,:,:,:] = image

        return {'video_x': new_video_x, 'video_label': video_label}

class ToTenor(object):
    """
    Convert ndarrays in sample to tensors.

    """

    def __call__(self, sample):
        video_x, video_label = sample['video_x'], sample['video_label']

        # swap color axis because
        # numpy image: [batch_size, D, H, W, C]
        # torch image: [batch_size, C, D, H, W]
        video_x = video_x.transpose((3,0,1,2))
        video_x = np.array(video_x)
        video_x /= 255
        video_label = [video_label]
        return {'video_x': torch.from_numpy(video_x).cuda(), 
                'video_label': torch.FloatTensor(video_label).cuda()}

class Normalize(object):
    def __init__(self, mean = 0.5, std = 0.5):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        video_x, video_label = sample['video_x'], sample['video_label']
        for i in range(video_x.size(1)):
            temp = video_x[:,i,:,:]
            temp = (temp - self.mean) / self.std
            video_x[:,i,:,:] = temp
        return {'video_x': video_x, 'video_label': video_label}

class UCF101_test(Dataset):
    def __init__(self, info_list, root_dir, transform = None):
        """
        Args:
            info_list (string): Path to the info list file with annotations.
            root_dir (string): Directory with all the video frames.
            transform (callable, optional): Optional transform to be applied
                                            on a sample.
    
        """

        self.landmarks_frame = pd.read_csv(info_list, delimiter = ' ', header = None)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, index):
        video_path = self.root_dir + '/' + self.landmarks_frame.iloc[index, 0]
        video_label = self.landmarks_frame.iloc[index, 1]
        videos = self.get_single_video_x(video_path)
        samples = []

        for video_x in videos:
            sample = {'video_x': video_x, 'video_label': video_label}
            if self.transform:
                sample = self.transform(sample)
            samples.append(sample)
        
        return samples

    def get_single_video_x(self, video_path):
        depth = D
        salsh_rows = video_path.split('.')
        video_jpgs_path = salsh_rows[0]
        data = pd.read_csv((video_jpgs_path + '/n_frames'), delimiter = ' ', header = None)
        frame_count = data[0][0]
        loop_num = int((frame_count - depth) / (depth / 2) + 1)
        video_x = np.zeros((depth,240,320,3))

        videos = []
        for image_start in range(loop_num):
            image_id = image_start * (depth / 2) + 1

            for i in range(depth):
                s = "%05d" % image_id
                image_name = 'image_' + s + '.jpg'
                image_path = video_jpgs_path + '/' + image_name
                temp_image = io.imread(image_path)
                video_x[i,:,:,:] = temp_image
                image_id += 1

            videos.append(video_x)

        return videos
