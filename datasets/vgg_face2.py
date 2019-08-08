#!/usr/bin/env python

import collections
import os

import numpy as np
import PIL.Image
import scipy.io
import torch
from torch.utils import data
import torchvision.transforms

class VGG_Faces2(data.Dataset):

    mean_bgr = np.array([91.4953, 103.8827, 131.0912])  # from resnet50_ft.prototxt

    def __init__(self, root, split='valid', transform=True,
                 horizontal_flip=False, upper=None):
        """
        :param root: dataset directory
        :param image_list_file: contains image file names under root
        :param id_label_dict: X[class_id] -> label
        :param split: train or valid
        :param transform: 
        :param horizontal_flip:
        :param upper: max number of image used for debug
        """
        assert os.path.exists(root), "root: {} not found.".format(root)
        self.root = root
        #assert os.path.exists(image_list_file), "image_list_file: {} not found.".format(image_list_file)
        #self.image_list_file = image_list_file
        self.split = split
        self._transform = transform
        #self.id_label_dict = id_label_dict
        self.horizontal_flip = horizontal_flip

        self.img_info = []

        self.img_info.append({
            'cid': "0",
            'img': root,
            'lbl': "1",
        })
        print("get the image "+ str(root))


        # with open(self.image_list_file, 'r') as f:
        #     for i, img_file in enumerate(f):
        #         img_file = img_file.strip()  # e.g. train/n004332/0317_01.jpg
        #         #class_id = img_file.split("/")[0]  # like n004332
        #         print("ID is " + "no class")
        #         #label = self.id_label_dict[class_id]
        #         self.img_info.append({
        #             'cid': "0",
        #             'img': img_file,
        #             'lbl': "1",
        #         })
        #         if i % 1000 == 0:
        #             print("processing: {} images for {}".format(i, self.split))
        #         if upper and i == upper - 1:  # for debug purpose
        #             break

    def __len__(self):
        return len(self.img_info)

    def __getitem__(self, index):
        info = self.img_info[index]
        img_file = info['img']

        img = PIL.Image.open(self.root)
        img = torchvision.transforms.Resize(256)(img)

        img = torchvision.transforms.CenterCrop(224)(img)
        if self.horizontal_flip:
            img = torchvision.transforms.functional.hflip(img)

        img = np.array(img, dtype=np.uint8)
        assert len(img.shape) == 3  # assumes color images and no alpha channel

        if self._transform:
            return self.transform(img)
        else:
            return img

    def transform(self, img):
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float32)
        img -= self.mean_bgr
        img = img.transpose(2, 0, 1)  # C x H x W
        img = torch.from_numpy(img).float()
        return img

    def untransform(self, img, lbl):
        img = img.numpy()
        img = img.transpose(1, 2, 0)
        img += self.mean_bgr
        img = img.astype(np.uint8)
        img = img[:, :, ::-1]
        return img, lbl

