import os
import sys
import numpy as np
import random
import open3d as o3d
from pyquaternion import Quaternion
from torch.utils.data import Dataset
import h5py

#index_to_label = np.array([12, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, -1, -1, 11], dtype='int32')
#label_to_index = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 0], dtype='int32')
#index_to_class = ['Void', 'Sky', 'Building', 'Road', 'Sidewalk', 'Fence', 'Vegetation', 'Pole', 'Car', 'Traffic Sign', 'Pedestrian', 'Bicycle', 'Lanemarking', 'Reserved', 'Reserved', 'Traffic Light']

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
    def __init__(self, root='data/pc', frames_per_clip=3, train=False):
        super(SegDataset, self).__init__()

        self.frames_per_clip = frames_per_clip
        self.train = train

        self.pcd = []
        self.center = []
        self.semantic = []
        for filename in ['test1_right.h5', 'test2_right.h5']:
            print(filename)
            with h5py.File(root+'/'+filename,'r') as f:
                self.pcd.append(np.array(f['pcd']))
                self.center.append(np.array(f['center']))
                self.semantic.append(np.array(f['semantic']))

    def __len__(self):
        leng = 89200
        return leng

    def read_training_data_point(self, index):

        idx = int(index / 100)
        data_id = index % 100
        frame_id = data_id * 3

        s = int(idx / 500)
        d = idx % 500

        pc = self.pcd[s][d][frame_id:frame_id+self.frames_per_clip]
        rgb = self.pcd[s][d][frame_id:frame_id+self.frames_per_clip]
        semantic = self.semantic[s][d][frame_id:frame_id+self.frames_per_clip]
        center_0 = self.center[s][d][frame_id]

        return pc, rgb, semantic, center_0

    def label_conversion(self, semantic):
        labels = []
        for i, s in enumerate(semantic):
            sem = s.astype('int32')
            label = index_to_label_vec_func(sem)

            labels.append(label)
        return labels

    def __getitem__(self, index):

        pc, rgb, semantic, center = self.read_training_data_point(index)
        
        label = self.label_conversion(semantic)
        label = np.array(label)
        
        rgb = np.swapaxes(rgb, 1, 2)

        return pc.astype(np.float32), rgb.astype(np.float32), label.astype(np.int64)


if __name__ == '__main__':
    datasets = SegDataset(root='/share/datasets/Seg_data_base_h5',train = False)