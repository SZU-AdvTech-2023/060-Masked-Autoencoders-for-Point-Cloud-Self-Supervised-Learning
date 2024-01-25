import os
import torch
import numpy as np
import torch.utils.data as data
from .io import IO
from .build import DATASETS
from utils.logger import *
import open3d as o3d
from utils import misc
@DATASETS.register_module()
class ShapeNet(data.Dataset):
    def __init__(self, config):
        self.data_root = config.DATA_PATH
        self.pc_path = config.PC_PATH
        self.subset = config.subset
        self.npoints = config.N_POINTS
        
        self.data_list_file = os.path.join(self.data_root, f'{self.subset}.txt')
        test_data_list_file = os.path.join(self.data_root, 'test.txt')
        
        self.sample_points_num = config.npoints
        self.whole = config.get('whole')

        print_log(f'[DATASET] sample out {self.sample_points_num} points', logger = 'ShapeNet-55')
        print_log(f'[DATASET] Open file {self.data_list_file}', logger = 'ShapeNet-55')
        with open(self.data_list_file, 'r') as f:
            lines = f.readlines()
        if self.whole:
            with open(test_data_list_file, 'r') as f:
                test_lines = f.readlines()
            print_log(f'[DATASET] Open file {test_data_list_file}', logger = 'ShapeNet-55')
            lines = test_lines + lines
        self.file_list = []
        for line in lines:
            line = line.strip()
            taxonomy_id = line.split('-')[0]
            model_id = line.split('-')[1].split('.')[0]
            self.file_list.append({
                'taxonomy_id': taxonomy_id,
                'model_id': model_id,
                'file_path': line
            })
        print_log(f'[DATASET] {len(self.file_list)} instances were loaded', logger = 'ShapeNet-55')

        self.permutation = np.arange(self.npoints)
    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc
    def points_to_o3d(self,points):

        cloud_o3d = o3d.geometry.PointCloud()

        cloud_o3d.points = o3d.utility.Vector3dVector(points[:, :3])

        return cloud_o3d  
    def calculate_normals(self,pc,threshold, radius_s, radius_l):
        cloud_o3d = self.points_to_o3d(pc[:, :3])

        cloud_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius_s, 64))

        normals = np.array(cloud_o3d.normals)

        cloud_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius_l, 64))

        normall = np.asarray(cloud_o3d.normals)

        don = (normals - normall) / 2

        removed = []

        don=np.linalg.norm(don,axis=-1)
        sort_don=np.argsort(don)[::-1]
        small_points=pc[sort_don[:128]]
        #print(small_points.shape)
        #small_points=pc[removed,:]
       # print(tosmall_points.shape)
        #small_points=misc.fps(torch.tensor(small_points).float().to(device).unsqueeze(0),128)
        #points_list=torch.cat([points_list,small_points],dim=0)
        return small_points
    def random_sample(self, pc, num):
        np.random.shuffle(self.permutation)
        pc = pc[self.permutation[:num]]
        return pc
        
    def __getitem__(self, idx):
        sample = self.file_list[idx]

        data = IO.get(os.path.join(self.pc_path, sample['file_path'])).astype(np.float32)

        data = self.random_sample(data, self.sample_points_num)
        data = self.pc_norm(data)
        key=self.calculate_normals(data,0.1,0.05,0.15)
        data = torch.from_numpy(data).float()
        key=torch.from_numpy(key).float()
        #print(key.size())
        return sample['taxonomy_id'], sample['model_id'], data,key

    def __len__(self):
        return len(self.file_list)