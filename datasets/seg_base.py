import os
import sys
import numpy as np
import random
import open3d as o3d
from pyquaternion import Quaternion
from torch.utils.data import Dataset
import h5py

# index_to_label = np.array([12, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, -1, -1, 11], dtype='int32')
# label_to_index = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 0], dtype='int32')
# index_to_class = ['Void', 'Sky', 'Building', 'Road', 'Sidewalk', 'Fence', 'Vegetation', 'Pole', 'Car', 'Traffic Sign', 'Pedestrian', 'Bicycle', 'Lanemarking', 'Reserved', 'Reserved', 'Traffic Light']

index_to_label = np.arange(0, 49, dtype='int32')
label_to_index = np.arange(0, 49, dtype='int32')
index_to_class = [str(i) for i in range(0, 49)]


# label_to_index = np.array([-1, -1, -1, 0, -1, 1, -1, -1, -1, 2, -1, 3, -1, -1, -1, 4, -1, 5, -1, 6, -1, 7, -1, 8, -1, 9, -1, 10, -1, -1, -1, -1, -1, 11, -1, 12, -1, 13, -1, 14, -1, 15, -1, -1, -1, 16, -1, 17, -1], dtype='int32')
# index_to_label = np.array([3, 5, 9, 11, 15, 17, 19, 21, 23, 25, 27, 33, 35, 37, 39, 41, 45, 47], dtype='int32')
# index_to_class = [str(i) for i in range(0, 49)]

def index_to_label_func(x):
    return index_to_label[x]


index_to_label_vec_func = np.vectorize(index_to_label_func)


class SegDataset(Dataset):
    def __init__(self, root='data/pc', frames_per_clip=3, num_points=8192, train=True):
        super(SegDataset, self).__init__()

        self.frames_per_clip = frames_per_clip
        self.train = train
        self.num_points = num_points

        self.pcd = []
        self.center = []
        self.semantic = []
        if self.train:
            for filename in ['train1.h5', 'train2.h5', 'train3.h5', 'train4.h5']:
                # for filename in ['train1.h5']:
                print(filename)
                with h5py.File(root + '/' + filename, 'r') as f:
                    self.pcd.append(np.array(f['pcd']))
                    self.center.append(np.array(f['center']))
                    self.semantic.append(np.array(f['semantic']))
        else:
            for filename in ['test1_right.h5', 'test2_right.h5']:
                print(filename)
                with h5py.File(root + '/' + filename, 'r') as f:
                    self.pcd.append(np.array(f['pcd']))
                    self.center.append(np.array(f['center']))
                    self.semantic.append(np.array(f['semantic']))
        # self.pcd = np.concatenate(self.pcd, axis=0)
        # self.center = np.concatenate(self.center,axis=0)
        # self.semantic = np.concatenate(self.semantic,axis=0)

    def __len__(self):
        leng = 0
        if self.train:
            leng = 2971 * 100
        else:
            leng = 892 * 100
        # leng = 750
        return leng

    def read_training_data_point(self, index):
        frame_idx = index % 100
        frame_id = frame_idx * 3

        if self.train:
            idx = int(index / 100)
            s = int(idx / 750)
            d = idx % 750
        else:
            idx = int(index / 100)
            s = int(idx / 500)
            d = idx % 500

        # pc = self.pcd[index][frame_id:frame_id+self.frames_per_clip]
        # rgb = self.pcd[index][frame_id:frame_id+self.frames_per_clip]
        # semantic = self.semantic[index][frame_id:frame_id+self.frames_per_clip]
        # center_0 = self.center[index][frame_id]
        # cho = np.array(range(0,30,3))
        pc = self.pcd[s][d][frame_id:int(frame_id + self.frames_per_clip)]
        rgb = self.pcd[s][d][frame_id:int(frame_id + self.frames_per_clip)]
        semantic = self.semantic[s][d][frame_id:int(frame_id + self.frames_per_clip)]
        center_0 = self.center[s][d][frame_id]

        return pc, rgb, semantic, center_0

    def augment(self, pc, center):
        flip = np.random.uniform(0, 1) > 0.5
        if flip:
            pc = pc - center
            jittered_data = np.clip(0.01 * np.random.randn(self.frames_per_clip, self.num_points, 3), -1 * 0.05, 0.05)
            jittered_data += pc
            pc = pc + center

        scale = np.random.uniform(0.8, 1.2)
        pc = (pc - center) * scale + center

        rot_axis = np.array([0, 1, 0])
        rot_angle = np.random.uniform(np.pi * -0.05, np.pi * 0.05)
        q = Quaternion(axis=rot_axis, angle=rot_angle)
        R = q.rotation_matrix

        pc = np.dot(pc - center, R) + center
        return pc

    def label_conversion(self, semantic):
        labels = []
        for i, s in enumerate(semantic):
            sem = s.astype('int32')
            label = index_to_label_vec_func(sem)

            labels.append(label)
        return labels

    def choice_to_num_points(self, pc, rgb, label):

        # shuffle idx to change point order (change FPS behavior)
        for f in range(self.frames_per_clip):
            idx = np.arange(pc[f].shape[0])
            choice_num = self.num_points
            if pc[f].shape[0] > choice_num:
                shuffle_idx = np.random.choice(idx, choice_num, replace=False)
            else:
                shuffle_idx = np.concatenate(
                    [np.random.choice(idx, choice_num - idx.shape[0]), np.arange(idx.shape[0])])
            pc[f] = pc[f][shuffle_idx]
            rgb[f] = rgb[f][shuffle_idx]
            label[f] = label[f][shuffle_idx]

        pc = np.stack(pc, axis=0)
        rgb = np.stack(rgb, axis=0)
        label = np.stack(label, axis=0)

        return pc, rgb, label

    def __getitem__(self, index):

        pc, rgb, semantic, center = self.read_training_data_point(index)

        label = self.label_conversion(semantic)
        label = np.array(label)

        if self.train:
            pc = self.augment(pc, center)

        rgb = np.swapaxes(rgb, 1, 2)
        # print(pc.shape)

        return pc.astype(np.float32), rgb.astype(np.float32), label.astype(np.int64)


if __name__ == '__main__':
    datasets = SegDataset(root='/share/datasets/Seg_data_base_h5', frames_per_clip=3, train=True)
