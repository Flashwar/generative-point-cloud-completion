import torch.utils.data as data
import numpy as np
import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import data_transforms
from .io import IO
import random
import os
import json
from .build import DATASETS
from utils.logger import *

@DATASETS.register_module()
class Catenary(data.Dataset):
    def __init__(self, config):
        self.partial_points_path = config.PARTIAL_POINTS_PATH
        self.complete_points_path = config.COMPLETE_POINTS_PATH
        self.category_file = config.CATEGORY_FILE_PATH
        self.npoints = config.N_POINTS
        self.subset = config.subset
        self.cars = config.CARS

        # Load the dataset indexing file
        self.dataset_categories = []
        with open(self.category_file) as f:
            self.dataset_categories = json.loads(f.read())
            if config.CARS:
                self.dataset_categories = [dc for dc in self.dataset_categories if dc['taxonomy_id'] == '02958343']

        self.file_list = self._get_file_list(self.subset)
        self.transforms = self._get_transforms(self.subset)

    def _get_transforms(self, subset):
        if subset == 'train':
            return data_transforms.Compose([{
                'callback': 'RandomSamplePoints',
                'parameters': {
                    'n_points': 2048
                },
                'objects': ['partial']
            }, {
                'callback': 'RandomMirrorPoints',
                'objects': ['partial', 'gt']
            },{
                'callback': 'ToTensor',
                'objects': ['partial', 'gt']
            }])
        else:
            return data_transforms.Compose([{
                'callback': 'RandomSamplePoints',
                'parameters': {
                    'n_points': 2048
                },
                'objects': ['partial']
            }, {
                'callback': 'ToTensor',
                'objects': ['partial', 'gt']
            }])

    def _get_file_list(self, subset):
        """Prepare file list for the dataset"""
        file_list = []
        for dc in self.dataset_categories:
            print_log('Collecting files of Taxonomy [ID=%s, Name=%s]' % (dc['taxonomy_id'], dc['taxonomy_name']), logger='CATENARYDATASET')
            samples = dc[subset]

            for s in samples:
                # Dynamisch die tatsächlichen Dateien im Partial-Ordner lesen
                partial_dir = os.path.dirname(
                    self.partial_points_path % (subset, dc['taxonomy_id'], s, 0)
                )
                if os.path.exists(partial_dir):
                    partial_files = sorted([
                        os.path.join(partial_dir, f)
                        for f in os.listdir(partial_dir)
                        if f.endswith('.pcd')
                    ])
                else:
                    partial_files = []

                file_list.append({
                    'taxonomy_id': dc['taxonomy_id'],
                    'model_id': s,
                    'partial_path': partial_files,
                    'gt_path': self.complete_points_path % (subset, dc['taxonomy_id'], s),
                })

        print_log('Complete collecting files of the dataset. Total files: %d' % len(file_list), logger='CATENARYDATASET')
        return file_list

    def __getitem__(self, idx):
        sample = self.file_list[idx]
        data = {}

        n_renderings = len(sample['partial_path'])
        rand_idx = random.randint(0, n_renderings - 1) if self.subset == 'train' else 0

        for ri in ['partial', 'gt']:
            file_path = sample['%s_path' % ri]
            if type(file_path) == list:
                file_path = file_path[rand_idx]
            data[ri] = IO.get(file_path).astype(np.float32)

        # Center the point clouds and normalize to unit sphere
        centroid = data['gt'].mean(axis=0)
        data['gt'] -= centroid
        data['partial'] -= centroid
        max_dist = np.max(np.sqrt(np.sum(data['gt'] ** 2, axis=1)))
        if max_dist > 0:
            data['gt'] /= max_dist
            data['partial'] /= max_dist

        if self.transforms is not None:
            data = self.transforms(data)

        return sample['taxonomy_id'], sample['model_id'], (data['partial'], data['gt'])

    def __len__(self):
        return len(self.file_list)