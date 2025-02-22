import collections
import os.path as osp
import numpy as np
from PIL import Image
import scipy.io
import torch
from torch.utils import data
import cv2
import copy
import random


class SegmentationData_BaseClass(data.Dataset):
    class_names = np.array([
        "road",
        "sidewalk",
        "building",
        "wall",
        "fence",
        "pole",
        "light",
        "sign",
        "vegetation",
        "terrain",
        "sky",
        "person",
        "rider",
        "car",
        "truck",
        "bus",
        "train",
        "motocycle",
        "bicycle",
    ])
    mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])

    def __init__(self, dset, root, split='train', transform=False, image_size=[1024, 512]):
        self.root = root
        self.image_size = image_size
        self.files = collections.defaultdict(list)

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        data_file = self.files[self.split][index]

        # Loading image and label
        img, lbl = self.image_label_loader(data_file['img'], data_file['lbl'], self.image_size, random_crop=True)
        img = img[:, :, ::-1]
        img -= self.mean_bgr
        img = img.transpose(2, 0, 1)

        if self.dset != 'cityscapes':
            lbl[lbl > 18] = -1

        img = torch.from_numpy(img.copy()).float()
        lbl = torch.from_numpy(lbl.copy()).long()

        return img, lbl

    def transform(self, img, lbl):
        """
        Module for implementing transformations to the input.
        Args:
            img: Image
            lbl: Label
        Returns:
            Transformed image and label
        """

        if self.dset != 'cityscapes':
            lbl = cv2.resize(lbl.astype(np.int32), tuple(self.image_size), cv2.INTER_CUBIC)
        lbl = np.array(lbl, dtype=np.int32)
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()

        return img, lbl

    def untransform(self, img):
        """
        Module for applying inverse transformation (removing mean normalization). This module
        is useful for visualization
        Args:
            img: Image (numpy nd array)
        Returns:
            Unransformed image (numpy nd array)
        """

        img = img.transpose(1, 2, 0)
        img += self.mean_bgr
        img = img.astype(np.uint8)
        img = img[:, :, ::-1]
        return img

    def transform_forD(self, img_orig, sz, resize=True, mean_add=True):
        """
        Module to transform the input as needed by the discriminator.
        Input to the discriminator is in the range [0, 1]
        Args:
            img_orig: Image
            sz: New size needed for discriminator
        Returns:
            Transformed image
        """

        if not resize and not mean_add:
            img = copy.deepcopy(img_orig)
            return img / 255.0
        else:
            img = copy.deepcopy(img_orig)
            img = img.numpy()
            img = img.squeeze().transpose((1, 2, 0))
            if mean_add:
                img += self.mean_bgr
            img = img[:, :, ::-1]

            if resize:
                img = img.astype(np.uint8)
                img = Image.fromarray(img).convert('RGB')
                img = img.resize((sz[0], sz[1]), Image.LANCZOS)
                in_ = np.array(img, dtype=np.float64)
            else:
                in_ = img

            in_ /= 255.0
            in_ = in_[:, :, ::-1].transpose((2, 0, 1))
            in_ = torch.from_numpy(in_.copy()).float()

            return in_

        """
        img = copy.deepcopy(img_orig)
        img = img.numpy()
        img = img.transpose((0, 2, 3, 1))
        img_new = np.zeros((img.shape[0], sz[1], sz[0], img.shape[3])).astype(img.dtype)
        for i in range(img.shape[0]):
            img[i] += self.mean_bgr
            img_new[i] = cv2.resize(img[i],sz)
            img_new[i] /= 255.0
        return img_new.transpose((0, 3, 1, 2))
        """

    def transform_label_forD(self, label_orig, sz):

        label = copy.deepcopy(label_orig.data.cpu().numpy())
        label[label == -1] = 19
        label = Image.fromarray(label.squeeze().astype(np.uint8))
        label = label.resize((sz[0], sz[1]), Image.NEAREST)
        label = np.array(label, dtype=np.int32)
        label[label > 18] = -1
        label = torch.from_numpy(label.copy()).long()
        label = label.cuda()
        return label

    def image_label_loader(self, img_path, label_path, data_size, random_crop=False):
        """
        Function for loading a single (image, label) pair
        Args:
            img_path: Image path to load
            label_path: Corresponding label path
            data_size: Required size. Aspect ratio needs to be 2:1 (Cityscapes aspect ratio)
            random_crop: Boolean variable to indicate if random crop needs to be performed. 
                         If False, centercrop is done
        Returns:
            Loaded image (numpy array)
        """

        if data_size[0] != data_size[1] * 2:
            raise ValueError('Specified aspect ratio not 2:1')

        im = Image.open(img_path).convert('RGB')
        label = Image.open(label_path)
        w = im.size[0]
        h = im.size[1]
        asp_rat = float(w) / float(h)

        if asp_rat > 2:
            wnew = h * 2
            hnew = h
            start_ind = random.randint(0, w - wnew)
            im = im.crop((start_ind, 0, wnew, hnew))
            label = label.crop((start_ind, 0, wnew, hnew))
        else:
            wnew = w
            hnew = w / 2
            start_ind = random.randint(0, h - hnew)
            im = im.crop((0, start_ind, wnew, hnew))
            label = label.crop((0, start_ind, wnew, hnew))

        im = im.resize((data_size[0], data_size[1]), Image.LANCZOS)
        label = label.resize((data_size[0], data_size[1]), Image.NEAREST)
        im_ = np.array(im, dtype=np.float64)
        label_ = np.array(label, dtype=np.int32)
        return im_, label_


class SYNTHIA(SegmentationData_BaseClass):

    def __init__(self, dset, root, split='train', transform=False, image_size=[1024, 512]):
        super(SYNTHIA, self).__init__(
            dset, root, split=split, transform=transform, image_size=image_size)

        self.dset = dset
        self.root = root
        self.filelist_path = osp.join(self.root, 'filelist')
        self.split = split
        self._transform = transform

        dataset_dir = osp.join(self.root, 'RAND_CITYSCAPES')
        # self.files = collections.defaultdict(list)

        if split == 'train':
            imgsets_file = open(osp.join(
                self.filelist_path, 'SYNTHIA_imagelist_train.txt'), 'r')
            label_file = open(osp.join(
                self.filelist_path, 'SYNTHIA_labellist_train.txt'), 'r')
        elif split == 'val':
            imgsets_file = open(osp.join(
                self.filelist_path, 'SYNTHIA_imagelist_val.txt'), 'r')
            label_file = open(osp.join(
                self.filelist_path, 'SYNTHIA_labellist_val.txt'), 'r')
        else:
            raise ValueError('Invalid split type. Should be train or val')

        for did, lid in zip(imgsets_file, label_file):
            img_file = osp.join(dataset_dir, '%s' % did.rstrip('\n'))
            lbl_file = osp.join(dataset_dir, '%s' % lid.rstrip('\n'))
            self.files[split].append({
                'img': img_file,
                'lbl': lbl_file,
            })

    def image_label_loader(self, img_path, label_path, data_size, random_crop=False):
        """
        Function for loading a single (image, label) pair
        Args:
            img_path: Image path to load
            label_path: Corresponding label path
            data_size: Required size. Aspect ratio needs to be 2:1 (Cityscapes aspect ratio)
            random_crop: Boolean variable to indicate if random crop needs to be performed. 
                         If False, centercrop is done
        Returns:
            Loaded image (numpy array)
        """

        if data_size[0] != data_size[1] * 2:
            raise ValueError('Specified aspect ratio not 2:1')

        im = Image.open(img_path).convert('RGB')
        label = Image.open(label_path)
        w = im.size[0]
        h = im.size[1]
        # random crop only for train
        if self.split == 'train':
            im_ = np.array(im, dtype=np.float64)
            label_ = np.array(label, dtype=np.int32)
            # randomizing the left corner point to select a random crop
            x_rand = np.random.randint(low=0, high=im_.shape[0] - data_size[1])
            y_rand = np.random.randint(low=0, high=im_.shape[1] - data_size[0])
            label_ = label_[x_rand:x_rand + data_size[1], y_rand:y_rand + data_size[0]]
            label_[label_ > 18] = -1
            im_ = im_[x_rand:x_rand + data_size[1], y_rand:y_rand + data_size[0], :]
        else:
            im = im.resize((data_size[0], data_size[1]), Image.LANCZOS)
            label = label.resize((data_size[0], data_size[1]), Image.NEAREST)
            im_ = np.array(im, dtype=np.float64)
            label_ = np.array(label, dtype=np.int32)
        return im_, label_


class GTA5(SegmentationData_BaseClass):

    def __init__(self, dset, root, split='train', transform=False, image_size=[1024, 512]):
        super(GTA5, self).__init__(
            dset, root, split=split, transform=transform, image_size=image_size)

        self.dset = dset
        self.root = root
        self.filelist_path = osp.join(self.root, 'filelist')
        self.split = split
        self._transform = transform

        dataset_dir = osp.join(self.root, 'GTA-5')
        # self.files = collections.defaultdict(list)

        print
        self.filelist_path
        if split == 'train':
            imgsets_file = open(osp.join(
                self.filelist_path, 'GTA5_imagelist_train.txt'), 'r')
            label_file = open(osp.join(
                self.filelist_path, 'GTA5_labellist_train.txt'), 'r')
        elif split == 'val':
            imgsets_file = open(osp.join(
                self.filelist_path, 'GTA5_imagelist_val.txt'), 'r')
            label_file = open(osp.join(
                self.filelist_path, 'GTA5_labellist_val.txt'), 'r')
        else:
            raise ValueError('Invalid split type. Should be train or val')

        for did, lid in zip(imgsets_file, label_file):
            img_file = osp.join(dataset_dir, '%s' % did.rstrip('\n'))
            lbl_file = osp.join(dataset_dir, '%s' % lid.rstrip('\n'))
            self.files[split].append({
                'img': img_file,
                'lbl': lbl_file,
            })

    def image_label_loader(self, img_path, label_path, data_size, random_crop=False):
        """
        Function for loading a single (image, label) pair
        Args:
            img_path: Image path to load
            label_path: Corresponding label path
            data_size: Required size. Aspect ratio needs to be 2:1 (Cityscapes aspect ratio)
            random_crop: Boolean variable to indicate if random crop needs to be performed. 
                         If False, centercrop is done
        Returns:
            Loaded image (numpy array)
        """

        if data_size[0] != data_size[1] * 2:
            raise ValueError('Specified aspect ratio not 2:1')

        im = Image.open(img_path).convert('RGB')
        label = Image.open(label_path)
        w = im.size[0]
        h = im.size[1]
        ar = w / (float(h))
        w_new = 1280
        h_new = int(w_new / float(ar))

        # random crop only for train
        if self.split == 'train':
            im = im.resize((w_new, h_new), Image.LANCZOS)
            label = label.resize((w_new, h_new), Image.LANCZOS)
            im_ = np.array(im, dtype=np.float64)
            label_ = np.array(label, dtype=np.int32)
            done = False
            while not done:
                # randomizing the left corner point to select a random crop
                x_rand = np.random.randint(low=0, high=im_.shape[0] - data_size[1])
                y_rand = np.random.randint(low=0, high=im_.shape[1] - data_size[0])
                tmp = label_[x_rand:x_rand + data_size[1], y_rand:y_rand + data_size[0]]

                min_val = tmp.min()
                # check bounds for the labels/dimensions of the cropped frame
                if min_val <= 18 and tmp.shape[0] == data_size[1] and tmp.shape[1] == data_size[0]:
                    done = True
                    tmp[tmp > 18] = -1
                    label_ = tmp
                if not done:
                    print
                    label_.sum()
            im_ = im_[x_rand:x_rand + data_size[1], y_rand:y_rand + data_size[0], :]
        else:
            im = im.resize((data_size[0], data_size[1]), Image.LANCZOS)
            label = label.resize((data_size[0], data_size[1]), Image.NEAREST)
            im_ = np.array(im, dtype=np.float64)
            label_ = np.array(label, dtype=np.int32)
        return im_, label_


class CityScapes(SegmentationData_BaseClass):

    def __init__(self, dset, root, split='train', transform=False, image_size=[1024, 512]):
        super(CityScapes, self).__init__(
            dset, root, split=split, transform=transform, image_size=image_size)

        self.dset = dset
        self.root = root
        self.filelist_path = osp.join(self.root, 'filelist')
        self.split = split
        self._transform = transform

        if split == 'train':
            imgsets_file = open(osp.join(
                self.filelist_path, 'cityscapes_imagelist_train.txt'), 'r')
            label_file = open(osp.join(
                self.filelist_path, 'cityscapes_labellist_train.txt'), 'r')
        elif split == 'val':
            imgsets_file = open(osp.join(
                self.filelist_path, 'cityscapes_imagelist_val.txt'), 'r')
            label_file = open(osp.join(
                self.filelist_path, 'cityscapes_labellist_val.txt'), 'r')
        else:
            raise ValueError('Invalid split type. Should be train or val')

        dataset_dir = osp.join(self.root, 'cityscapes')
        if split == 'train':
            img_dir = osp.join(dataset_dir, 'leftImg8bit/train')
            gt_dir = osp.join(dataset_dir, 'gtFine/train')
        elif split == 'val':
            img_dir = osp.join(dataset_dir, 'leftImg8bit/val')
            gt_dir = osp.join(dataset_dir, 'gtFine/val')

        for did, lid in zip(imgsets_file, label_file):
            img_file = osp.join(img_dir, '%s' % did.rstrip('\n'))
            lbl_file = osp.join(gt_dir, '%s' % lid.rstrip('\n'))
            self.files[self.split].append({'img': img_file, 'lbl': lbl_file})

    def image_label_loader(self, img_path, label_path, data_size, random_crop=False):
        """
        Function for loading a single (image, label) pair
        Args:
            img_path: Image path to load
            label_path: Corresponding label path
            data_size: Required size. Aspect ratio needs to be 2:1 (Cityscapes aspect ratio)
            random_crop: Boolean variable to indicate if random crop needs to be performed. 
                         If False, centercrop is done
        Returns:
            Loaded image (numpy array)
        """

        if data_size[0] != data_size[1] * 2:
            raise ValueError('Specified aspect ratio not 2:1')

        im = Image.open(img_path).convert('RGB')
        label = Image.open(label_path)

        # random crop only for train
        if self.split == 'train':
            im_ = np.array(im, dtype=np.float64)
            label_ = np.array(label, dtype=np.int32)
            # randomizing the left corner point to select a random crop
            x_rand = np.random.randint(low=0, high=im_.shape[0] - 50 - data_size[1])
            y_rand = np.random.randint(low=0, high=im_.shape[1] - 50 - data_size[0])  # buffer of 50
            label_ = label_[x_rand:x_rand + data_size[1], y_rand:y_rand + data_size[0]]
            im_ = im_[x_rand:x_rand + data_size[1], y_rand:y_rand + data_size[0], :]
        else:
            im = im.resize((data_size[0], data_size[1]), Image.LANCZOS)
            im_ = np.array(im, dtype=np.float64)
            label_ = np.array(label, dtype=np.int32)

        return im_, label_
