from __future__ import print_function
import datetime
import os
import time
import sys
import numpy as np
import torch
import torch.utils.data
from torch.utils.data.dataloader import default_collate
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms

import utils

from scheduler import WarmupMultiStepLR

from datasets.seg_test import *
import models.seg_p4_base as Models
# import models.seg_pptr_base as Models     #test for p4 or pptr


def evaluate(model, criterion, data_loader, device, print_freq):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    total_loss = 0
    total_correct = 0
    total_pred_class = [0] * 49
    total_correct_class = [0] * 49
    total_class = [0] * 49

    with torch.no_grad():
        for pc1, rgb1, label1 in metric_logger.log_every(data_loader, print_freq, header):
            pc1, rgb1 = pc1.to(device), rgb1.to(device)
            output1 = model(pc1, rgb1).transpose(1, 2)
            # loss1 = criterion(output1, label1)
            loss1 = criterion(output1, label1.to(device))
            loss1 = torch.mean(loss1)
            label1 = label1.numpy().astype(np.int32)
            output1 = output1.cpu().numpy()
            output1 = output1[:,2:,:,:]
            pred1 = np.argmax(output1, 1)+2 # BxTxN
            correct1 = np.sum(pred1 == label1)
            total_correct += correct1

            for c in range(49):
                total_pred_class[c] += np.sum(((pred1==c) | (label1==c)))
                total_correct_class[c] += np.sum((pred1==c) & (label1==c))
                total_class[c] += np.sum((label1==c))

            metric_logger.update(loss=loss1.item())

    skiplist = [0, 1]

    ACCs = []
    tcc = 0
    tc = 0
    for c in range(49):
        print(c, total_class[c])
        if c in skiplist:
            continue
        if total_class[c] < 0.1:
            continue
        acc = total_correct_class[c] / float(total_class[c])
        tcc += total_correct_class[c]
        tc += total_class[c]
        print('eval acc of %s:\t %f'%(index_to_class[label_to_index[c]], acc))
        print("total_class:", total_class[c])
        print("total_pred_class:", total_pred_class[c])
        print("total_correct_class:", total_correct_class[c])
        ACCs.append(acc)
    if len(ACCs) > 0:
        print(' * Eval accuracy: %f'% (np.mean(np.array(ACCs))))
        print("ex: ", tcc / float(tc))

    IoUs = []
    tcc = 0
    tpc = 0
    for c in range(49):
        if c in skiplist:
            continue
        if total_class[c] < 0.1:
            continue
        iou = total_correct_class[c] / float(total_pred_class[c])
        tcc += total_correct_class[c]
        tpc += total_pred_class[c]
        print('eval mIoU of %s:\t %f'%(index_to_class[label_to_index[c]], iou))
        IoUs.append(iou)
    if len(IoUs) > 0:
        print("IoUs:",len(IoUs))
        print(' * Eval mIoU:\t %f'%(np.mean(IoUs)))
    return np.mean(IoUs)

def main(args):

    if args.output_dir:
        utils.mkdir(args.output_dir)

    print(args)
    print("torch version: ", torch.__version__)
    print("torchvision version: ", torchvision.__version__)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda')

    print("Loading data")

    st = time.time()

    dataset = SegDataset(root='/datasets/Seg_data',train=False)

    print("Creating data loaders")

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=False)

    print("Creating model")
    
    Model = getattr(Models, args.model)
    model = Model(radius=args.radius, nsamples=args.nsamples, num_classes=49)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        pre_state = checkpoint['model'] 
        # for name in pre_state.keys():
        #     print(name)
        update_dict = {k: v for k, v in pre_state.items() if k.startswith("module.conv") or k.startswith("module.transformer") or k.startswith("module.deconv") or k.startswith("module.outconv")} 
        for name in update_dict.keys():
            print(name)
        net_state_dict = model.state_dict()
        for name in net_state_dict.keys():
            print(name)
        net_state_dict.update(update_dict)
        model.load_state_dict(net_state_dict)

    criterion_test = nn.CrossEntropyLoss(reduction='none')

    print("Start training")
    best_iou = 0
    start_time = time.time()
    
    evaluate(model, criterion_test, data_loader, device=device, print_freq=args.print_freq)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Transformer Model Training')

    parser.add_argument('--data-path', default='', help='data path')
    parser.add_argument('--label-weight', default='', help='training label weights')
    
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    # change model name
    # parser.add_argument('--model', default='PrimitiveTransformer', type=str, help='model')
    parser.add_argument('--model', default='P4Transformer', type=str, help='model')
    # input
    parser.add_argument('--clip-len', default=3, type=int, metavar='N', help='number of frames per clip') ##############
    parser.add_argument('--num-points', default=8192, type=int, metavar='N', help='number of points per frame')
    # P4D
    parser.add_argument('--radius', default=0.9, type=float, help='radius for the ball query')
    parser.add_argument('--nsamples', default=32, type=int, help='number of neighbors for the ball query')
    # training
    parser.add_argument('-b', '--batch-size', default=24, type=int)
    parser.add_argument('-j', '--workers', default=10, type=int, metavar='N', help='number of data loading workers (default: 16)')
    # output
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='', type=str, help='path where to save')

    parser.add_argument('--resume', default='', help='resume from checkpoint')

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
