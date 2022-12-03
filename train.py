import os
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import sys
from models.resnet20 import ResNet as resnet_cifar
import pandas as pd
import argparse
import csv
from torch.optim.lr_scheduler import MultiStepLR
from train.dataLoader import DataLoader
from train.dataLoader import TinyImageNetDataset
from summaries import TensorboardSummary
from models.vgg_cifar import vgg16
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from models.inception_v3 import inceptionv3
from models.densenet import Network as DenseNet
from models.WRN import Network as WRN
from models.resnet18 import ResNet18
from models.densenet_new import DenseNet

# parameters setting
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--name', default='test_model', help='filename to output best model') #save output
parser.add_argument('--p', default=0.1,type=float,help="percentage of neuron to be cut")
parser.add_argument('--dropway', default='no',help="way to drop ")
parser.add_argument('--dataset', default='cifar-10',help="datasets")
parser.add_argument('--depth', default=20,type=int,help="depth of resnet model")
parser.add_argument('--gpu-ids', type=str, default='1,2',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
parser.add_argument('--batch_size', default=64,type=int, help='batch size')
parser.add_argument('--epoch', default=200,type=int, help='epoch')
parser.add_argument('--cutout', default=0,type=int, help='using cutout')
parser.add_argument('--exp_dir',default='./',help='dir for tensorboard')
parser.add_argument('--model',default='resnet',help='model to train')
parser.add_argument('--blocksize',default=3,type=int,help='block size')


args = parser.parse_args()

if os.path.exists(args.exp_dir):
    print ('Already exist')
summary = TensorboardSummary(args.exp_dir)
tb_writer = summary.create_summary()

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    #best_model = model.state_dic()
    best_acc = 0.0
    best_train_acc = 0.0

    for epoch in range(num_epochs):
        print('-'*10)
        print('Epoch {}/{}'.format(epoch,num_epochs-1))

        #each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)
            else:
                model.train(False)

            running_loss = 0.0
            running_corrects = 0.0

            # change tensor to variable(including some gradient info)
            # use variable.data to get the corresponding tenso
            train_time = 0
            for data in dataloaders[phase]:
                #782 batch,batch size= 64
                inputs, labels = data
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                #zero the parameter gradients
                optimizer.zero_grad()
                # print (inputs.shape)

                #forward
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs.data, 1)

                if phase == 'train':
                    loss.backward()
                    # print('backward')
                    optimizer.step()
                    train_time += 1

                y = labels.data
                running_loss += loss.item()
                running_corrects += torch.sum(preds == y)

            epoch_loss = running_loss /dataset_sizes[phase]
            epoch_acc = float(running_corrects) /dataset_sizes[phase]

            print('%s Loss: %.4f top1 Acc:%.4f'%(phase,epoch_loss,epoch_acc))
            if phase == 'train':
                tb_writer.add_scalar('train/total_loss_epoch', epoch_loss, epoch)
                tb_writer.add_scalar('train/acc_epoch', epoch_acc, epoch)
                if best_train_acc < epoch_acc:
                    best_train_acc = epoch_acc

            if phase == 'val':
                tb_writer.add_scalar('val/total_loss_epoch', epoch_loss, epoch)
                tb_writer.add_scalar('val/acc_epoch', epoch_acc, epoch)

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model = model.state_dict()



    cost_time = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(cost_time//60,cost_time%60))
    print ('Best Train Acc is {:.4f}'.format(best_train_acc))
    print ('Best Val Acc is {:.4f}'.format(best_acc))

    model.load_state_dict(best_model)
    return model, cost_time, best_acc, best_train_acc

def log(exp_dir):
    f = open(os.path.join(exp_dir,'log.txt'),'w')
    msg = ''
    msg = msg + ('DataSets: '+args.dataset+'\n')
    msg = msg + ('ResNet Depth: '+str(args.depth)+'\n')
    msg = msg + ('DropWay: '+args.dropway+'\n')
    msg = msg + ('Best Acc: {:.4f}\n'.format(best_acc))
    msg = msg + ('\n')
    f.write(msg)
    f.close()



if __name__ == '__main__':
    print ('DataSets: '+args.dataset)
    print ('Model:', args.model)
    print ('ResNet Depth: '+str(args.depth))
    print ('DropWay: '+args.dropway)
    print ('P:', args.p)
    print('Block Size:', args.blocksize)


    loader = DataLoader(args.dataset,
                        batch_size=args.batch_size,
                        cutout=args.cutout)
    dataloaders, dataset_sizes = loader.load_data()
    # torch.backends.cudnn.enabled = False

    num_classes = 10
    if args.dataset == 'cifar-10':
        num_classes = 10
    if args.dataset == 'cifar-100':
        num_classes = 100
    if args.dataset == 'svhn':
        num_classes = 10
    if args.dataset == 'tiny-imagenet':
        num_classes = 200

    
    if args.model == 'resnet':
        if args.depth == 18:
            model = ResNet18(num_classes=num_classes, p=args.p, dropway=args.dropway)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01,
                                        momentum=0.9, nesterov=True, weight_decay=5e-4)
            scheduler = MultiStepLR(optimizer, milestones=[50, 100, 150],
                                        gamma=0.1)
            print('resnet 18')
        else:
            if args.dataset == 'svhn':
                model = resnet_cifar(depth=args.depth,
                                         num_classes=num_classes,
                                         dropway=args.dropway,
                                         p=args.p,
                                         block_size=args.blocksize)
                optimizer = torch.optim.SGD(model.parameters(), lr=0.01,
                                                momentum=0.9, nesterov=True, weight_decay=5e-4)
                scheduler = MultiStepLR(optimizer, milestones=[80, 120], gamma=0.1)
            else:
                model = resnet_cifar(depth=args.depth,
                                    num_classes=num_classes,
                                    dropway=args.dropway,
                                    p=args.p,
                                    block_size=args.blocksize)
                optimizer = torch.optim.SGD(model.parameters(), lr=0.1,
                                            momentum=0.9, nesterov=True, weight_decay=1e-4)
                scheduler = MultiStepLR(optimizer, milestones=[80, 120, 160], gamma=0.1)
        if args.model == 'vgg':
            model = vgg16()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True, weight_decay=1e-4)
            scheduler = MultiStepLR(optimizer, milestones=[80, 120, 160], gamma=0.1)
        if args.model == 'inception':
            model = inceptionv3(num_classes=num_classes,dropway=args.dropway,p=args.p)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True, weight_decay=1e-4)
            scheduler = MultiStepLR(optimizer, milestones=[80, 120, 160], gamma=0.1)
        if args.model == 'densenet':
            if args.dropway == 'SCD':
                block_type = 'scd_bottleneck'
            elif args.dropway == 'CCD':
                block_type = 'ccd_bottleneck'
            elif args.dropwar == 'dropblock':
                block_type = 'dropblock_bottleneck'
            else:
                block_type = 'bottleneck'
            config = {
                "arch": "densenet",
                "depth": 100,
                "block_type": "bottleneck",
                "growth_rate": 12,
                "drop_rate": 0,
                "compression_rate": 0.5,
                "n_classes":10,
                "input_shape":(args.batch_size,3,32,32),
                "dropway":args.dropway,
                "p":args.p
            }
            model = DenseNet(config)
            # model = DenseNet(efficient=True)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True, weight_decay=1e-4)
            scheduler = MultiStepLR(optimizer, milestones=[150, 225], gamma=0.1)
        if args.model == 'wrn':
            config = {
                        "arch": "wrn",
                        "base_channels": 16,
                        "depth": 28,
                        "widening_factor": 10,
                        "drop_rate": 0,
                         "n_classes": 10,
                         "input_shape": (args.batch_size, 3, 32, 32),
                         "dropway": args.dropway,
                         "p": args.p
                    }
            model = WRN(config)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True, weight_decay=5e-4)
            scheduler = MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)

        # define loss and optimizer
        criterion = nn.CrossEntropyLoss()

    # define loss and optimizer
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')
        model = torch.nn.DataParallel(model, device_ids=args.gpu_ids)
        # patch_replication_callback(model)
        model = model.cuda()
    model,cost_time,best_acc,best_train_acc = train_model(model=model,
                                                          optimizer=optimizer,
                                                          criterion=criterion,
                                                          scheduler=scheduler,
                                                          num_epochs=args.epoch)

    exp_name = '%s%d dataset: %s dropway: %s drop_prop %.2f batchsize: %d epoch: %d cutout: %d bestValAcc: %.4f bestTrainAcc: %.4f \n' % (
    args.model,args.depth, args.dataset, args.dropway,args.p,args.batch_size, args.epoch,args.cutout,best_acc,best_train_acc)
    # log(cost_time,best_acc)
    if os.path.exists(args.exp_dir+'/checkpoints') == False:
        os.mkdir(args.exp_dir+'/checkpoints')
    torch.save(model.state_dict(), args.exp_dir+'/checkpoints/model.pt')
    log(args.exp_dir)

