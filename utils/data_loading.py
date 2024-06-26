import logging
import os
import numpy as np
import torch
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


def readFlow(fn):
    """ Read .flo file in Middlebury format"""
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None, None
        else:
            w = np.fromfile(f, np.int32, count=1)[0]
            h = np.fromfile(f, np.int32, count=1)[0]
            data = np.fromfile(f, np.float32, count=2*w*h)
            flow = np.resize(data, (h, w, 2))
            # valid = (np.abs(flow[..., 0]) < 1000) & (np.abs(flow[..., 1]) < 1000)  # add valid check
            # return flow, valid.astype(np.float32)
            return flow


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, flow_dir: str, scale: float = 1.0):
        self.images_dir = Path(images_dir)
        self.flow_dir = Path(flow_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale

        # self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]
        self.ids = []
        for root, _, files in os.walk(images_dir):
            for file in files:
                if file.endswith('.png') and not file.endswith('-checkpoint.png'):
                    full_path = os.path.join(root, splitext(file)[0])
                    relative_path = os.path.relpath(full_path, images_dir)
                    self.ids.append(relative_path)
        
        # use 5 images to test
        # self.ids = self.ids[:5]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        logging.info(f'Creating dataset with {len(self.ids)} examples')
        logging.info('Scanning mask files to determine unique values')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    # ignore first
    def preprocess(mask_values, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)

        if is_mask:
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i

            return mask

        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img / 255.0

            return img

    def __getitem__(self, idx):
        image_id = self.ids[idx]
        image_path = self.images_dir / (image_id + '.png')
        flow_path = self.flow_dir / (image_id + '.flo')

        
        if not image_path.is_file():
            print(f"Image file not found for ID {image_id}")
            return None 
        if not flow_path.is_file():
            print(f"Flow file not found for ID {image_id}")
            return None

        
        image = Image.open(image_path)
        image = BasicDataset.preprocess(None, image, self.scale, is_mask=False)
        # image = torch.tensor(image).permute(2, 0, 1)  # convert HWC -> CHW
        image = torch.tensor(image)
        # flow, valid = readFlow(flow_path)
        flow = readFlow(flow_path)
        
        if flow is None:
            print(f"Failed to read flow file for ID {image_id}")
            return None  


        return {
            'image': image.contiguous(),
            'flow': flow,
            # 'valid': torch.from_numpy(valid).float(),
        }
    
    
# process KITTI dataset
def readFlowKITTI(filename):
    """Read KITTI optical flow map from .png file"""
    flow = cv2.imread(filename, cv2.IMREAD_UNCHANGED | cv2.IMREAD_COLOR)
    flow = flow[:, :, ::-1].astype(np.float32)  # convert BGR to RGB
    flow = (flow[:, :, :2] - 32768) / 64.0
    return flow

class KITTI(Dataset):
    def __init__(self, images_dir: str, flow_dir: str):
        self.images_dir = Path(images_dir)
        self.flow_dir = Path(flow_dir)

        self.ids = []
        for file in os.listdir(images_dir):
            if file.endswith('_10.png'):
                file_id = file.replace('_10.png', '')
                self.ids.append(file_id)
        
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

   
    def __getitem__(self, idx):
        image_id = self.ids[idx]
        image_path = self.images_dir / f'{image_id}_10.png'
        flow_path = self.flow_dir / f'{image_id}_10.png'

        if not image_path.is_file() or not flow_path.is_file():
            logging.error(f"Files not found for ID {image_id}: {image_path}, {flow_path}")
            return None 

        image = Image.open(image_path)
        image = self.preprocess(image, is_mask=False)
        image = torch.tensor(image)
        flow = readFlowKITTI(str(flow_path))  # 确保路径为字符串

        return {
            'image': image.contiguous(),
            'flow': flow,
        }

    @staticmethod
    def preprocess(pil_img, is_mask=False):
        img = np.array(pil_img)
        
        if img.ndim == 2:
            img = img[np.newaxis, ...]
        else:
            img = img.transpose((2, 0, 1))

        if img.max() > 1:
            img = img / 255.0  # Normalize if not already
        
        return img
