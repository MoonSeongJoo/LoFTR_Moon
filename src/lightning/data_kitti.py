import os
import math
from collections import abc
from loguru import logger
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm
from os import path as osp
from pathlib import Path
from joblib import Parallel, delayed

import pytorch_lightning as pl
from torch import distributed as dist
from torch.utils.data import (
    Dataset,
    DataLoader,
    ConcatDataset,
    DistributedSampler,
    RandomSampler,
    dataloader
)

# from src.utils.augment import build_augmentor
from src.utils.dataloader import get_local_split
from src.utils.misc import tqdm_joblib
from src.utils import comm
# from src.datasets.megadepth import MegaDepthDataset
# from src.datasets.scannet import ScanNetDataset
# from src.datasets.sampler import RandomConcatSampler
from src.datasets.DatasetLidarCamera_Ver9_3 import DatasetLidarCameraKittiOdometry

class KittiDataModule(pl.LightningDataModule):
    """ 
    For distributed training, each training process is assgined
    only a part of the training scenes to reduce memory overhead.
    """
    def __init__(self, args, config):
        super().__init__()

        # 1. data config
        # training and validating
        self.data_path = "/mnt/sgvrnas/sjmoon/kitti/kitti_odometry"
        self.max_r = 20.0
        self.max_t = 1.5
        # 2. dataset config
        # general options
        # self.min_overlap_score_test = config.DATASET.MIN_OVERLAP_SCORE_TEST  # 0.4, omit data with overlap_score < min_overlap_score
        # self.min_overlap_score_train = config.DATASET.MIN_OVERLAP_SCORE_TRAIN
        # self.augment_fn = build_augmentor(config.DATASET.AUGMENTATION_TYPE)  # None, options: [None, 'dark', 'mobile']

        # 3.loader parameters
        self.use_reflectance = False
        self.val_sequence = '06'
        self.batch_size = 10  # 120
        self.num_worker = 10

        # misc configurations
        # self.parallel_load_data = getattr(args, 'parallel_load_data', False)
        self.seed = config.TRAINER.SEED  # 66

    def setup(self, stage=None):
        """
        Setup train / val / test dataset. This method will be called by PL automatically.
        Args:
            stage (str): 'fit' in training phase, and 'test' in testing phase.
        """        
        
        self.trainset = DatasetLidarCameraKittiOdometry(self.data_path, max_r = self.max_r, max_t = self.max_t, 
                                                        split='train' , use_reflectance = self.use_reflectance,
                                                        val_sequence= self.val_sequence)
        self.validset = DatasetLidarCameraKittiOdometry(self.data_path, max_r = self.max_r, max_t = self.max_t,
                                                        split='val', use_reflectance = self.use_reflectance,
                                                        val_sequence= self.val_sequence)
        train_dataset_size = len(self.trainset)
        val_dataset_size = len(self.validset)
        print('Number of the train dataset: {}'.format(train_dataset_size))
        print('Number of the val dataset: {}'.format(val_dataset_size))

    def train_dataloader(self):
        """ Build training dataloader for kitti. """
        dataloader = DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True,
                                num_workers=self.num_worker, collate_fn = self.merge_inputs,
                                drop_last =False,
                                pin_memory=True) #  worker_init_fn=self.seed,
        print(len(dataloader))
        return dataloader
    
    def val_dataloader(self):
        """ Build validation dataloader for kitti. """
        dataloader = DataLoader(self.validset, batch_size=self.batch_size, shuffle=False,
                                num_workers=self.num_worker, collate_fn = self.merge_inputs,
                                drop_last =False,
                                pin_memory=True) #   ,worker_init_fn=self.seed,
        print(len(dataloader))
        return dataloader
    
    def merge_inputs(self, queries):
        point_clouds = []
        imgs = []
        reflectances = []
        corrs =[]
        pc_rotated =[]
        depth_imgs = []
        depth0 =[]
        depth1 =[]
    #     returns = {key: default_collate([d[key] for d in queries]) for key in queries[0]
    #                if key != 'point_cloud' and key != 'rgb' and key != 'reflectance' }
        returns = {key: default_collate([d[key] for d in queries]) for key in queries[0]
            if key != 'point_cloud' and key != 'image0' and key != 'image1' and key != 'reflectance' and key != 'corrs' and key != 'pc_rotated' 
            and key != 'depth0' and key != 'depth1'}
        for input in queries:
            point_clouds.append(input['point_cloud'])
            imgs.append(input['image0'])
            depth_imgs.append(input['image1'])
            depth0.append(input['depth0'])
            depth1.append(input['depth1'])
            corrs.append(input['corrs'])
            pc_rotated.append(input['pc_rotated'])
            if 'reflectance' in input:
                reflectances.append(input['reflectance'])
        returns['point_cloud'] = point_clouds
        returns['image0'] = imgs
        returns['image1'] = depth_imgs
        returns['corrs'] = corrs
        returns['pc_rotated'] = pc_rotated
        if len(reflectances) > 0:
            returns['reflectance'] = reflectances
        return returns

def _build_dataset(dataset: Dataset, *args, **kwargs):
    return dataset(*args, **kwargs)
