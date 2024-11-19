# Python 2-3 compatible
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
# Packages
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from augmix import AugMixDataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision

import numpy as np

from copy import deepcopy
from tqdm import tqdm
import argparse
import uuid
import os
import importlib

import datetime
import time

from model.utils import validate, accuracy

def load_cifar10_c(dataroot, batch_size, workers, corruption='gaussian_noise', level=1):
    NORM = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    test_size = 10000
    test_transforms = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(*NORM)])

    test_raw = np.load(dataroot + f'/CIFAR-10-C/{corruption}.npy')
    test_raw = test_raw[(level-1)*test_size : level*test_size] # slicing
    test_set = datasets.CIFAR10(root=dataroot,
                            train=False,
                            download=True, 
                            transform=test_transforms)
    test_set.data = test_raw
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False,
                                        num_workers=workers, drop_last=True)

    return test_set, test_loader

def load_cifar10(dataroot='../data', no_jsd=True):
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=dataroot, train=True, download=True, transform=transform_train)
    trainset = AugMixDataset(trainset, preprocess, no_jsd=no_jsd)

    testset = torchvision.datasets.CIFAR10(
        root=dataroot, train=False, download=True, transform=preprocess)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    return trainloader, testloader

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Continual Domain Adaptation')
    ## data args
    parser.add_argument('--dataroot', default='/data/kien/',help='path where data is located')

    ## model args
    parser.add_argument('--pretrained', type=str)
    parser.add_argument('--erm_pretrained', type=str)
    parser.add_argument('--model', type=str, default='meta_tta', help='model to train')
    parser.add_argument('--ss', default=0.1, type=float)
    parser.add_argument('--meta_w', default=0.1, type=float)

    ## steps args
    parser.add_argument('--inner_steps', type=int, default=1)
    parser.add_argument('--n_meta', type=int, help='# of meta steps', default=1)
    parser.add_argument('--grad_step_online', type=int, default=1)

    parser.add_argument('--batch_size', type=int, default=200, help='batch size')
    parser.add_argument('--adapt_batch_size', type=int, default=32, help='batch size during adaptation')
    parser.add_argument('--opt', type=str, default='sgd')
    parser.add_argument('--lr', type=float, default=0.1, help='SGD learning rate')
    parser.add_argument('--inner_w', type=float, default=0.1, help='inner loop learning rate weight')
    parser.add_argument('--online_lr', type=float, default=0.001, help='Online SGD learning rate')

    parser.add_argument('--cuda', action='store_true', help='Use GPU?')
    parser.add_argument('--seed', type=int, default=0, help='random seed')

    ## logging args
    parser.add_argument('--save_path', type=str, default='results/', help='save models at the end of training')

    parser.add_argument('--group_norm', type=int, default=0)

    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--nesterov', action='store_true')
    parser.add_argument('--weight_decay', default=0.0, type=float, help='weight decay (default: 0.0)')
    parser.add_argument('--depth', default=26, type=int)
    parser.add_argument('--width', default=1, type=int)

    # epochs
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--erm_epochs', default=0, type=int)
    parser.add_argument('--epochs', default=75, type=int)
    parser.add_argument('--milestones', default=[20, 40, 50, 60, 70])
    parser.add_argument('--outf', default='../checkpoints')

    args = parser.parse_args()
    print(args)

    # seed
    if int(args.seed) > -1:
        torch.cuda.manual_seed_all(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # GPU or CPU
    device = 'cuda' if torch.cuda.is_available else 'cpu'

    # fname and stuffs
    if args.pretrained:
        uid = args.pretrained.split(".")[0].split("_")[0]
    else:
        uid = uuid.uuid4().hex[:8]
    start_time = time.time()
    fname = args.model + '_' + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    fname += '_' + uid + '_' \
            + str(args.ss) + 'ss_' \
            + str(args.online_lr) + 'onlinelr_' \
            + str(args.lr) + 'lr_' \
            + str(args.inner_steps) + 'inner_' \
            + str(args.adapt_batch_size) + 'bs_' \
            + str(args.group_norm) + 'group_'

    fname = os.path.join(args.save_path, fname)
    logfile = open(fname + '.log', 'w')
    logfile.write(str(args))
    logfile.write('\n')

    # domains for adaptation
    '''
        CIFAR-10-C has 1 .npy file for each corruption type on the testset.
        The shape is (50000, 32, 32, 3) for each .npy data.
        => (#level * 10000, 32, 32, 3).
    '''
    common_corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise',
                        'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
                        'snow', 'frost', 'fog', 'brightness',
                        'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
    # Define the model
    Model = importlib.import_module('model.' + args.model)
    model = Model.Net(n_outputs=10, exp_name=uid, args=args)

    cudnn.benchmark = True

    # Pre-train on CIFAR-10
    train_loader, val_loader = load_cifar10() # we can use large batchsize here. 32/64/128
    model.train()

    if args.erm_pretrained:
        from model.utils import test
        _, acc, _ = test(val_loader, model.net, device)
        print(acc*100.)
        print("=> Pre-training...")
        model.pretrain(train_loader, val_loader, logfile)
    elif args.pretrained:
        from model.utils import test
        _, acc, _ = test(val_loader, model.net, device)
        print(acc*100.)
    else:
        print("=> Pre-training...")
        model.pretrain(train_loader, val_loader, logfile)

    current_task = -1
    
    criterion = nn.CrossEntropyLoss()

    # ACC
    accs = [] # accs on all domains at the end of adaptation

    # BWT
    tt = [] # accs on target domain t after adapting to t
    Nt = [] # accs on prev target domains after adapting to the last domain

    # FWT
    Tt = [] # accs on target domain t when adapting solely source to t
    errs = []

    model.configure_bn_params()

    for didx, corruption in enumerate(common_corruptions): # adaptation
        _, test_loader = load_cifar10_c(args.dataroot, args.adapt_batch_size, workers=4, corruption=corruption, level=5)

        current_task += 1
        info = current_task
        desc = 'Training task {}'.format(current_task)

        preds = []
        labels = []

        for epoch in range(1):
            for x, y in tqdm(test_loader, ncols=69, desc=desc):
                x = x.float().cuda()
                y = y.long().cuda()
                pred = model.adapt(x)

                labels.append(y.detach().cpu())
                preds.append(pred.detach().cpu())
        
        acc = accuracy(torch.cat(preds), torch.cat(labels), topk=(1,))[0]
        errs.append(100. - acc)
        print(f"Corruption {corruption} - Error: {100. - acc}")

        if didx != len(common_corruptions) - 1: # if not last target domain
            top1 = validate(test_loader, model.net, criterion, 0)
            tt.append(top1)
        else:
            # BWT, test model trained on last domain on previous target domains
            for cor in common_corruptions[:-1]:
                _, test_loader = load_cifar10_c(args.dataroot, args.adapt_batch_size, workers=4,
                                corruption=cor, level=5)
                top1 = validate(test_loader, model.net, criterion, 0)
                print(f"Corruption {cor} - Acc: {top1:.4f}")
                Nt.append(top1)
                accs.append(top1)

            # FWT
            _, test_loader = load_cifar10_c(args.dataroot, args.adapt_batch_size, workers=4,
                                corruption=common_corruptions[-1], level=5)
            top1 = validate(test_loader, model.net, criterion, 0)
            print(f"Corruption {common_corruptions[-1]} - Acc: {top1:.4f}")
            accs.append(top1)

    # Online Error
    for corruption, err in zip(common_corruptions, errs):
        logfile.write(f"{corruption}: \t\t\t {err} \n")

    print(f"Avg: {str(np.asarray(errs).mean())}")
    logfile.write(f"Avg: {str(np.asarray(errs).mean())}")

    # ACC metrics
    print(f'ACC: {sum(accs) / len(accs)}')
    logfile.write(f'ACC: {sum(accs) / len(accs)}\n')

    # BWT metrics
    tt = np.array(tt)
    Nt = np.array(Nt)
    bwt = np.mean(Nt - tt)
    print(f'BWT: {bwt}')
    logfile.write(f'BWT: {bwt}\n')

    # FWT metrics, adapting solely from source to each target
    for cor in common_corruptions[1:]:
        model.reset()
        current_task = 0
        _, test_loader = load_cifar10_c(args.dataroot, args.adapt_batch_size, workers=4,
                        corruption=cor, level=5)
        for x, y in tqdm(test_loader, ncols=69, desc=desc):
            x = x.float().cuda()
            y = y.long().cuda()
            pred = model.adapt(x)
        top1 = validate(test_loader, model.net, criterion, 0)
        print(f"Corruption {cor} - Acc: {top1:.4f}")
        Tt.append(top1)
    
    accs = np.array(accs)
    Tt = np.array(Tt)
    fwt = np.mean(accs[1:] - Tt)
    print(f'FWT: {fwt}')
    logfile.write(f'FWT: {fwt}\n')

    # Calculate time spent
    time_spent = time.time() - start_time
    print(f"Total Time: {time_spent // 3600 % 24} hrs {time_spent // 60 % 60} mins")
    logfile.close()