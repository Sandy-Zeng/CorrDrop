import torchvision.datasets as datasets
import sys
from cutout import Cutout
import numpy.random as npr
from adaptive_cutout import AdaptiveCutout
from PIL import Image

def default_loader(path):
    return Image.open(path).convert('RGB')
import os
import torch
import numpy as np


class DataLoader():
    def __init__(self,dataset,batch_size,cutout):
        self.dataset = dataset
        self.batch_size = batch_size
        self.cutout = cutout

    def load_data(self):
        data_dir = './data'
        data_transforms = {
            'train': transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
                    # transforms.Normalize([0.5, 0.5, 0.5], [1, 1, 1])
            ]),
            'val': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
                # transforms.Normalize([0.5, 0.5, 0.5], [1, 1, 1])
            ])
            ,
            'vis': transforms.Compose([
                 transforms.ToTensor(),
            ])

        }
        if self.cutout == 1:
            data_transforms['train'].transforms.append(Cutout(n_holes=2, length=10))
            data_transforms['vis'].transforms.append(Cutout(n_holes=2, length=10))
        if self.cutout == 2:
            data_transforms['train'].transforms.append(AdaptiveCutout(p=0.5))
        if self.dataset == 'cifar-10':
            data_train = datasets.CIFAR10(root=data_dir,
                                          transform=data_transforms['train'],
                                          train=True,
                                          download=True)
            data_test = datasets.CIFAR10(root=data_dir,
                                         transform=data_transforms['val'],
                                         train=False,
                                         download=True)
            # if self.noise_labels_percentage > 0:
            #     data_train = introduce_noise(data_train,self.noise_labels_percentage)
        if self.dataset == 'cifar-100':
            data_train = datasets.CIFAR100(root=data_dir,
                                          transform=data_transforms['train'],
                                          train=True,
                                          download=True)

            data_test = datasets.CIFAR100(root=data_dir,
                                         transform=data_transforms['val'],
                                         train=False,
                                         download=True)
        if self.dataset == 'svhn':
            train_transformers = transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                                         ])
            if self.cutout == 1:
                train_transformers.transforms.append(Cutout(n_holes=2, length=10))
            data_train = datasets.SVHN(root=data_dir,
                                       split='train',
                                       download=True,
                                       transform=train_transformers)
            data_test = datasets.SVHN(root=data_dir,
                                      split='test',
                                      download=True,
                                      transform=transforms.Compose([
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                      ]))
        if self.dataset == 'tiny-imagenet':
            data_dir = './data/tiny-imagenet-200'
            train_dir = os.path.join(data_dir, 'train')
            val_dir = './data/tiny-imagenet-200/val/images'
            channel_means = (0.442, 0.442, 0.442)
            channel_stdevs = (0.278, 0.278, 0.278)
            normalize = transforms.Normalize(mean=channel_means,
                                             std=channel_stdevs)
            train_transformers = transforms.Compose([
                transforms.RandomCrop(64, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
            val_transformers = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
            if self.cutout == 1:
                train_transformers.transforms.append(Cutout(n_holes=2, length=10))
            data_train = datasets.ImageFolder(train_dir, train_transformers)
            data_test = datasets.ImageFolder(val_dir, val_transformers)

        image_datasets = {'train': data_train, 'val': data_test}
        # change list to Tensor as the input of the models
        dataloaders = {}
        dataloaders['train'] = torch.utils.data.DataLoader(image_datasets['train'],
                                                           batch_size=self.batch_size,
                                                           shuffle=True)
        dataloaders['val'] = torch.utils.data.DataLoader(image_datasets['val'],
                                                         batch_size=self.batch_size,
                                                         shuffle=False)
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
        return dataloaders, dataset_sizes


class TinyImageNetDataset(torch.utils.data.Dataset):
    def __init__(self, root, data_list, transform = None, loader = default_loader):
        # root: your_path/TinyImageNet/
        # data_list: your_path/TinyImageNet/train.txt etc.
        images = []
        labels = open(data_list).readlines()
        for line in labels:
            items = line.strip('\n').split()
            img_name = items[0]

            # test list contains only image name
            test_flag = True if len(items) == 1 else False
            label = None if test_flag == True else np.array(int(items[1]))

            if os.path.isfile(os.path.join(root, img_name)):
                images.append((img_name, label))
            else:
                print(os.path.join(root, img_name) + 'Not Found.')

        self.root = root
        self.images = images
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        img_name, label = self.images[index]
        img = self.loader(os.path.join(self.root, img_name))
        raw_img = img.copy()
        if self.transform is not None:
            img = self.transform(img)
        return (img, label) if label is not None else img

    def __len__(self):
        return len(self.images)

if __name__ == '__main__':
    dataloader = DataLoader(dataset='CUB_Bird',batch_size=128)
    loaders,sizes = dataloader.load_data()
    print (sizes)



def introduce_noise(train_dataset,noise_percent):
    nlabels = len(train_dataset.train_labels)
    nlabels_to_change = int(noise_percent * nlabels / 100)
    nclasses = len(np.unique(train_dataset.train_labels))
    print('flipping ' + str(nlabels_to_change) + ' labels')

    # Randomly choose which labels to change, get indices
    labels_inds_to_change = npr.choice(
        np.arange(nlabels), nlabels_to_change, replace=False)

    # Flip each of the randomly chosen labels
    for l, label_ind_to_change in enumerate(labels_inds_to_change):
        # Possible choices for new label
        label_choices = np.arange(nclasses)

        # Get true label to remove it from the choices
        true_label = train_dataset.train_labels[label_ind_to_change]

        # Remove true label from choices
        label_choices = np.delete(
            label_choices,
            true_label)  # the label is the same as the index of the label

        # Get new label and relabel the example with it
        noisy_label = npr.choice(label_choices, 1)
        train_dataset.train_labels[label_ind_to_change] = noisy_label[0]
    return train_dataset


