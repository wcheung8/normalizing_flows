import os
import torch
import numpy as np
import scipy.misc as m
import cv2
from torch.utils import data
from torchvision import transforms

from ptsemseg.utils import recursive_glob


class fishyscapesLoader(data.Dataset):
    """cityscapesLoader

    https://www.cityscapes-dataset.com

    Data is derived from CityScapes, and can be downloaded from here:
    https://www.cityscapes-dataset.com/downloads/

    Many Thanks to @fvisin for the loader repo:
    https://github.com/fvisin/dataset_loaders/blob/master/dataset_loaders/images/cityscapes.py
    """

    def __init__(
            self,
            split="train",
            is_transform=True,
            img_size=(512, 1024),
            augmentations=None,
            img_norm=True,
            version="cityscapes", **kwargs
    ):
        """__init__

        :param root:
        :param split:
        :param is_transform:
        :param img_size:
        :param augmentations
        """
        print('hi')

    def __len__(self):
        """__len__"""
        return 1

    def __getitem__(self, index):
        """__getitem__

        :param index:
        """
        import bdlb

        fs = bdlb.load(benchmark="fishyscapes")
        # automatically downloads the dataset
        data = fs.get_dataset()
        img = []
        lbl = []
        for i, blob in enumerate(data.take(10)):
            img.append(blob['image_left'])
            lbl.append(blob['mask'])
            print(blob['image_left'])
        return img, lbl


if __name__ == "__main__":
    import torchvision
    import matplotlib.pyplot as plt

    augmentations = Compose([Scale(2048), RandomRotate(10), RandomHorizontallyFlip(0.5)])

    local_path = "/datasets01/cityscapes/112817/"
    dst = cityscapesLoader(local_path, is_transform=True, augmentations=augmentations)
    bs = 4
    trainloader = data.DataLoader(dst, batch_size=bs, num_workers=0)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        import pdb;

        pdb.set_trace()
        imgs = imgs.numpy()[:, ::-1, :, :]
        imgs = np.transpose(imgs, [0, 2, 3, 1])
        f, axarr = plt.subplots(bs, 2)
        for j in range(bs):
            axarr[j][0].imshow(imgs[j])
            axarr[j][1].imshow(dst.decode_segmap(labels.numpy()[j]))
        plt.show()
        a = raw_input()
        if a == "ex":
            break
        else:
            plt.close()
