import datetime
import math
import os
import gc
import time

import numpy as np
import torch
from torch.autograd import Variable

import utils
import tqdm

class Extractor(object):

    def __init__(self, cuda, model, val_loader1, val_loader2, log_file, flatten_feature=True, print_freq=1):
        """
        :param cuda:
        :param model:
        :param val_loader:
        :param log_file: log file name. logs are appended to this file.
        :param feature_dir:
        :param flatten_feature:
        :param print_freq:
        """
        self.cuda = cuda

        self.model = model
        self.val_loader1 = val_loader1
        self.val_loader2 = val_loader2

        self.log_file = log_file

        self.flatten_feature = flatten_feature
        self.print_freq = print_freq

        self.timestamp_start = datetime.datetime.now()


    def print_log(self, log_str):
        with open(self.log_file, 'a') as f:
            f.write(log_str + '\n')


    def extract_hang(self):
        #batch_time = utils.AverageMeter()

        self.model.eval()
        #end = time.time()
        with torch.no_grad():
            for batch_idx, imgs in enumerate(self.val_loader1):     ##########1 is output

                gc.collect()

                imgs = imgs.cuda()
                imgs = Variable(imgs)
                output1 = self.model(imgs)  # N C H W torch.Size([1, 1, 401, 600])
                if self.flatten_feature:
                    output1 = output1.view(output1.size(0), -1)
                output1 = output1.data.cpu().numpy()
                print("The output feature is " + str(output1[0]))

        with torch.no_grad():
            for batch_idx, imgs in enumerate(self.val_loader2):       ########2 is groundtruth

                gc.collect()

                imgs = imgs.cuda()
                imgs = Variable(imgs)
                output2 = self.model(imgs)  # N C H W torch.Size([1, 1, 401, 600])
                if self.flatten_feature:
                    output2 = output2.view(output2.size(0), -1)
                output2 = output2.data.cpu().numpy()
                print("The groundtruth feature is " + str(output2[0]))

        simi = np.dot(output1[0], output2[0])/(np.linalg.norm(output1[0]) * (np.linalg.norm(output2[0])))
        print("simi is")
        print(simi)
