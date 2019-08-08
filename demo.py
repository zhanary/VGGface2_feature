#!/usr/bin/env python

import argparse
import os
import sys
import torch
import torch.nn as nn

import datasets
import models.resnet as ResNet
import models.senet as SENet

from extractor import Extractor
import utils
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"



configurations = {
    1: dict(
        max_iteration=1000000,
        lr=1.0e-1,
        momentum=0.9,
        weight_decay=0.0,
        gamma=0.1, # "lr_policy: step"
        step_size=1000000, # "lr_policy: step"
        interval_validate=1000,
    ),
}

def get_parameters(model, bias=False):
    for k, m in model._modules.items():
        if k == "fc" and isinstance(m, nn.Linear):
            if bias:
                yield m.bias
            else:
                yield m.weight

N_IDENTITY = 8631  # the number of identities in VGGFace2 for which ResNet and SENet are trained

def main():
    parser = argparse.ArgumentParser("PyTorch Face Recognizer")
    parser.add_argument('--cmd', type=str,  default='extract', help='train, test or extract')
    parser.add_argument('--arch_type', type=str, default='resnet50_ft', help='model type',
                        choices=['resnet50_ft', 'senet50_ft', 'resnet50_scratch', 'senet50_scratch'])
    parser.add_argument('--log_file', type=str, default='log_file.txt', help='log file')

    parser.add_argument('-c', '--config', type=int, default=1, choices=configurations.keys(),
                        help='the number of settings and hyperparameters used in training')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--weight_file', type=str, default='resnet50_ft_weight.pkl', help='weight file')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--horizontal_flip', action='store_true', 
                        help='horizontally flip images specified in test_img_list_file')
    args = parser.parse_args()
    print(args)

    log_file = args.log_file

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    cuda = torch.cuda.is_available()

    print("torch.backends.cudnn.version: {}".format(torch.backends.cudnn.version()))

    torch.manual_seed(1337)

    torch.cuda.manual_seed(1337)

    # 0. id label map

    # 1. data loader

    ###########
    img_output = "./nirvis/P002.jpg"            ########the image
    img_groundtruth = "./nirvis/P004_9.jpg"
    ###########


    kwargs = {'num_workers': args.workers, 'pin_memory': True} if cuda else {}


    f_img_output = datasets.VGG_Faces2(img_output, split='valid', horizontal_flip=args.horizontal_flip)        #return  transform(img), label, img_file, class_id
    f_img_groundtruth = datasets.VGG_Faces2(img_groundtruth, split='valid', horizontal_flip=args.horizontal_flip)

    val_loader_output = torch.utils.data.DataLoader(f_img_output, batch_size=1, shuffle=False, **kwargs)
    val_loader_groundtruth = torch.utils.data.DataLoader(f_img_groundtruth, batch_size=1, shuffle=False, **kwargs)

    #val_loader = torch.utils.data.DataLoader

    # 2. model
    include_top = True if args.cmd != 'extract' else False
    if 'resnet' in args.arch_type:
        model = ResNet.resnet50(num_classes=N_IDENTITY, include_top=include_top)
    else:
        model = SENet.senet50(num_classes=N_IDENTITY, include_top=include_top)
    # print(model)

    utils.load_state_dict(model, args.weight_file)

    model = model.cuda()

    extractor = Extractor(
        cuda=cuda,
        model=model,
        val_loader1=val_loader_output,
        val_loader2=val_loader_groundtruth,

        log_file=log_file,

        flatten_feature=True,
        print_freq=1,
    )
    extractor.extract_hang()


if __name__ == '__main__':

    main()
