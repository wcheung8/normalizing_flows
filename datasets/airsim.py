import matplotlib

matplotlib.use('Agg')

import os
import torch
import numpy as np
import scipy.misc as m
import glob
import cv2
import time
import matplotlib.pyplot as plt
import copy
from random import shuffle
import random
from torch.utils import data
import yaml
from tqdm import tqdm
import pickle
from ptsemseg.degredations import *

random.seed(42)


def label_region_n_compute_distance(i, path_tuple):
    begin = path_tuple[0]
    end = path_tuple[1]

    # computer distance
    distance = ((begin[0] - end[0]) ** 2 + (begin[1] - end[1]) ** 2) ** (0.5)

    # label region
    if begin[0] <= -400 or end[0] < -400:
        region = 'suburban'
    else:
        if begin[1] >= 300 or end[1] >= 300:
            region = 'shopping'
        else:
            region = 'skyscraper'

    # update tuple
    path_tuple = (i,) + path_tuple + (distance, region,)

    return path_tuple


class airsimLoader(data.Dataset):
    name2color = {"person": [[135, 169, 180]],
                  # "ground":      [[106,16,239]],
                  "sidewalk": [[242, 107, 146]],
                  "road": [[156, 198, 23], [43, 79, 150]],
                  "sky": [[209, 247, 202]],
                  "pole": [[249, 79, 73], [72, 137, 21], [45, 157, 177], [67, 266, 253], [206, 190, 59]],
                  "building": [[161, 171, 27], [61, 212, 54], [151, 161, 26]],
                  "car": [[153, 108, 6]],
                  "bus": [[190, 225, 64]],
                  "truck": [[112, 105, 191]],
                  "vegetation": [[29, 26, 199], [234, 21, 250], [145, 71, 201], [247, 200, 111]],
                  }

    name2id = {"person": 1,
               "sidewalk": 2,
               "road": 3,
               "sky": 4,
               "pole": 5,
               "building": 6,
               "car": 7,
               "bus": 8,
               "truck": 9,
               "vegetation": 10}

    id2name = {i: name for name, i in name2id.items()}

    splits = ['train', 'val', 'recal']
    image_modes = ['scene', 'depth', 'segmentation']

    # weathers = ['8camera_fog_000_dense', '8camera_rain_dense']

    # cam_pos = ['back_lower',  'back_upper']

    cam_pos = ['back', ]
    # cam_pos = [ 'back_lower',  'back_upper',
    #             'front_lower', 'front_upper',
    #             'left_lower',  'left_upper',
    #             'right_lower', 'right_upper']

    all_edges = [
        ((0, 0), (16, -74)),
        ((16, -74), (-86, -78)),
        ((-86, -78), (-94, -58)),
        ((-94, -58), (-94, 24)),
        ((-94, 24), (-143, 24)),
        ((-143, 24), (-219, 24)),
        ((-219, 24), (-219, -68)),
        ((-219, -68), (-214, -127)),
        ((-214, -127), (-336, -132)),
        ((-336, -132), (-335, -180)),
        ((-335, -180), (-216, -205)),
        ((-216, -205), (-226, -241)),
        ((-226, -241), (-240, -252)),
        ((-240, -252), (-440, -260)),
        ((-440, -260), (-483, -253)),
        ((-483, -253), (-494, -223)),
        ((-494, -223), (-493, -127)),
        ((-493, -127), (-441, -129)),
        ((-441, -129), (-443, -222)),
        ((-443, -222), (-339, -221)),
        ((-339, -221), (-335, -180)),
        ((-335, -180), (-336, -132)),
        ((-336, -132), (-214, -127)),
        ((-214, -127), (-219, -68)),
        ((-219, -68), (-219, 24)),
        ((-219, 24), (-248, 24)),
        ((-248, 24), (-302, 24)),
        ((-302, 24), (-337, 24)),
        ((-337, 24), (-593, 25)),
        ((-593, 25), (-597, -128)),
        ((-597, -128), (-597, -220)),
        ((-597, -220), (-748, -222)),
        ((-748, -222), (-744, -128)),
        ((-744, -128), (-746, 24)),
        ((-746, 24), (-744, -128)),
        ((-744, -128), (-597, -128)),
        ((-597, -128), (-593, 25)),
        ((-593, 25), (-746, 24)),
        ((-746, 24), (-832, 27)),
        ((-832, 27), (-804, 176)),
        ((-804, 176), (-747, 178)),
        ((-747, 178), (-745, 103)),
        ((-745, 103), (-696, 104)),
        ((-696, 104), (-596, 102)),
        ((-596, 102), (-599, 177)),
        ((-599, 177), (-747, 178)),
        ((-747, 178), (-599, 177)),
        ((-599, 177), (-597, 253)),
        ((-597, 253), (-599, 177)),
        ((-599, 177), (-596, 102)),
        ((-596, 102), (-593, 25)),
        ((-593, 25), (-337, 24)),
        ((-337, 24), (-337, 172)),
        ((-337, 172), (-332, 251)),
        ((-332, 251), (-337, 172)),
        ((-337, 172), (-221, 172)),
        ((-221, 172), (-221, 264)),
        ((-221, 264), (-221, 172)),
        ((-221, 172), (-219, 90)),
        ((-219, 90), (-219, 24)),
        ((-219, 24), (-219, 90)),
        ((-219, 90), (-221, 172)),
        ((-221, 172), (-148, 172)),
        ((-148, 172), (-130, 172)),
        ((-130, 172), (-57, 172)),
        ((-57, 172), (-57, 194)),
        ((-57, 194), (20, 192)),
        ((20, 192), (20, 92)),
        ((20, 92), (21, 76)),
        ((21, 76), (66, 22)),
        ((66, 22), (123, 28)),
        ((123, 28), (123, 106)),
        ((123, 106), (123, 135)),
        ((123, 135), (176, 135)),
        ((176, 135), (176, 179)),
        ((176, 179), (210, 180)),
        ((210, 180), (210, 107)),
        ((210, 107), (216, 26)),
        ((216, 26), (118, 21)),
        ((118, 21), (118, 2)),
        ((118, 2), (115, -33)),
        ((115, -33), (100, -62)),  #
        ((100, -62), (89, -70)),
        ((89, -70), (62, -76)),
        ((62, -76), (28, -76)),
        ((28, -76), (16, -74)),
        ((16, -74), (14, -17)),
        ((14, -17), (0, 0)),

        ((-494, -223), (-597, -220)),
        ((-597, -128), (-493, -127)),
        ((-493, -127), (-493, 25)),
        ((-441, -129), (-441, 25)),
        ((-336, -132), (-337, 24)),

        ((14, -17), (66, 22)),
        ((66, 22), (21, 76)),
        ((21, 76), (-24, 25)),
        ((-338, 172), (-332, 251)),
        ((-221, 172), (-221, 264)),
        ((20, 192), (20, 255)),
        ((176, 179), (176, 274)),
        ((-599, 177), (-597, 253)),

        ((-597, 253), (-443, 253)),
        ((-443, 253), (-332, 251)),
        ((-332, 251), (-221, 264)),
        ((-221, 264), (-211, 493)),
        ((-211, 493), (-129, 493)),
        ((-129, 493), (23, 493)),
        ((23, 493), (20, 274)),
        ((20, 274), (20, 255)),
        ((20, 255), (-57, 255)),
        ((-57, 255), (20, 255)),
        ((20, 255), (20, 274)),
        ((20, 274), (114, 274)),
        ((114, 274), (176, 267)),
        ((176, 267), (176, 348)),
        ((176, 348), (180, 493)),
        ((180, 493), (175, 660)),
        ((175, 660), (23, 646)),
        ((23, 646), (-128, 646)),
        ((-128, 646), (-134, 795)),
        ((-134, 795), (-130, 871)),
        ((-130, 871), (20, 872)),
        ((20, 872), (23, 798)),
        ((23, 798), (175, 795)),
        ((175, 795), (23, 798)),
        ((23, 798), (-134, 795)),
        ((-134, 795), (-128, 676)),
        ((-128, 676), (-129, 493)),
        ((-129, 493), (-211, 493)),
        ((-211, 493), (-129, 493)),
        ((-129, 493), (23, 493)),
        ((23, 493), (23, 646)),
        ((23, 646), (23, 798)),
        ((23, 798), (20, 872)),

    ]

    split_subdirs = {}

    # split_subdirs['train'] = [ 
    #                         "0_0__-94_58",
    #                         "123_-135__216_-26",
    #                         "-130_-255__-57_-255",
    #                         "-143_39__-219_68",
    #                         "176_-355__250_-180",
    #                         "-302_-24__-309_-172",
    #                         "-309_-172__-219_-264",
    #                         "-57_-255__88_-202",
    #                         "88_-202__20_-355",
    #                         "-94_58__-143_39",]

    # split_subdirs['val'] = [
    #                         "20_-355__176_-355",
    #                         "-219_-264__-130_-255",
    #                         "-219_68__-302_-24",
    #                         "250_-180__123_-135",]

    ignore_index = 0

    mean_rgbd = {
        # "airsim": [103.939, 116.779, 123.68, 120.00],
        # "airsim": [21,22,21,45]
        "airsim": [41.454376, 46.093113, 42.958637, 4.464941, 5.1877136, 167.58365]
    }  # pascal mean for PSPNet and ICNet pre-trained model

    std_rgbd = {
        # "airsim": [8,7,8,36],
        "airsim": [37.94737, 37.26296, 36.74846, 22.874805, 28.264046, 39.39389]
    }

    def __init__(
            self,
            root='/home/wcheung8/pytorch-semseg/datasets/airsim_data_async',
            split="train",
            subsplits=None,
            is_transform=False,
            img_size=(512, 512),
            scale_quantity=1.0,
            augmentations=None,
            img_norm=True,
            version="airsim",
    ):
        """__init__

        :param root:
        :param split:
        :param is_transform:
        :param img_size:
        :param augmentations 
        """

        self.dataset_div = self.divide_region_n_train_val_test()

        self.split_subdirs = self.generate_image_path(self.dataset_div)

        self.root = root
        self.split = split
        self.subsplits = subsplits
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.img_norm = img_norm
        self.n_classes = 11
        self.img_size = (img_size if isinstance(img_size, tuple) else (img_size, img_size))
        self.mean = np.array(self.mean_rgbd[version])
        self.std = np.array(self.std_rgbd[version])

        # split: train/val image_modes
        # self.imgs = {s:{image_mode:[] for image_mode in self.image_modes} for s in self.splits}
        self.imgs = {s: {c: {image_mode: [] for image_mode in self.image_modes} for c in self.cam_pos} for s in self.splits}
        self.dgrd = {s: {c: {image_mode: [] for image_mode in self.image_modes} for c in self.cam_pos} for s in self.splits}

        # print(self.splits,self.subsplits,self.split_subdirs[split])

        k = 0
        n = 0
        for split in self.splits:  # [train, val]
            for subsplit in self.subsplits:  # []

                print("Processing trajectories in {} {}".format(split, subsplit))
                for subdir in tqdm(self.split_subdirs[split]):  # [trajectory ]

                    if len(subsplit.split("__")) == 2:
                        condition = subsplit.split("__")[0]
                        degradation = subsplit.split("__")[1]
                    else:
                        condition = subsplit
                        degradation = None

                    for file_path in sorted(glob.glob(os.path.join(root, 'scene', condition, subdir, self.cam_pos[0], '*.png'), recursive=True)):

                        ext = file_path.replace(root + "/scene/", '')
                        env = ext.split("/")[1]
                        file_name = ext.split("/")[3]

                        list_of_all_cams_n_modal = [os.path.exists(os.path.join(root, modal, condition, subdir, cam, file_name)) for modal in self.image_modes for cam in self.cam_pos]
                        n = n + 1
                        # print(list_of_all_cams_n_modal)
                        if all(list_of_all_cams_n_modal):
                            k = k + 1
                            img_col = []

                            for comb_modal in self.image_modes:
                                img_row_list = []
                                for comb_cam in self.cam_pos:
                                    file_path = os.path.join(root, comb_modal, condition, subdir, comb_cam, file_name)
                                    # img = m.imread(file_path)

                                    self.imgs[split][comb_cam][comb_modal].append(file_path)
                                    self.dgrd[split][comb_cam][comb_modal].append(degradation)
                    '''
                    if all([os.path.exists(os.path.join(root,image_mode,ext)) for image_mode in self.image_modes]):
                        if subsplit is None or (not subsplit is None and subsplit==env):
                            [self.imgs[split][image_mode].append(os.path.join(root,image_mode,ext)) for image_mode in self.image_modes]
                    '''

        print('scene_back_image num', n)
        print('valid sample pairs', k)

        print("{} {}: Found {} Images".format(self.split, self.subsplits, len(self.imgs[self.split][self.cam_pos[0]][self.image_modes[0]])))
        if scale_quantity != 1.0:
            for cam_pos in self.cam_pos:
                for image_mode in self.image_modes:
                    self.imgs[self.split][cam_pos][image_mode] = self.imgs[self.split][cam_pos][image_mode][::int(1 / scale_quantity)]
            print("{} {}: Reduced by {} to {} Images".format(self.split, self.subsplits, scale_quantity, len(self.imgs[self.split][self.cam_pos[0]][self.image_modes[0]])))

        # self.dataset_statistics()
        # exit()

    def dataset_statistics(self):

        print("=" * 20)
        print("Running Dataset Statistics")
        print("=" * 20)

        print("Splits:    {}".format(self.split))
        print("Positions: {}".format(", ".join(list(self.imgs[self.split].keys()))))
        print("Modes:     {}".format(", ".join(list(self.imgs[self.split][self.cam_pos[0]].keys()))))

        savefile = "{}_dataset_statistics.p".format(self.split)
        savefile2 = "{}_pixels.p".format(self.split)

        rgb_mean = []
        rgb_std = []
        d_mean = []
        d_std = []
        for index in tqdm(range(int(1.0 * len(self.imgs[self.split][self.cam_pos[0]][self.image_modes[0]])))):
            input_list, lbl_list = self.__getitem__(index)
            img_list, aux_list = input_list['rgb'], input_list['d']
            # shape (batch_size, 3, height, width)
            numpy_image = torch.stack(img_list, 0).numpy()
            numpy_depth = torch.stack(aux_list, 0).numpy()

            # shape (3,)
            batch_mean = np.mean(numpy_image, axis=(0, 2, 3))
            batch_std = np.std(numpy_image, axis=(0, 2, 3), ddof=1)

            rgb_mean.append(batch_mean)
            rgb_std.append(batch_std)

            # shape (3,)
            batch_mean = np.mean(numpy_depth, axis=(0, 2, 3))
            batch_std = np.std(numpy_depth, axis=(0, 2, 3), ddof=1)

            d_mean.append(batch_mean)
            d_std.append(batch_std)

        # shape (num_iterations, 3) -> (mean across 0th axis) -> shape (3,)
        rgb_mean = np.array(rgb_mean).mean(axis=0)
        rgb_std = np.array(rgb_std).mean(axis=0)
        # shape (num_iterations, 3) -> (mean across 0th axis) -> shape (3,)
        d_mean = np.array(d_mean).mean(axis=0)
        d_std = np.array(d_std).mean(axis=0)

        print("rgb: mean - {}, std - {}".format(rgb_mean, rgb_std))
        print("d: mean - {}, std - {}".format(d_mean, d_std))

        # if os.path.isfile(savefile):
        if False:
            pixel_stats = pickle.load(open(savefile, "rb"))
            pixel_dump = pickle.load(open(savefile2, "rb"))

        else:
            pixel_stats = {p: {n: [] for n in self.name2id} for p in self.cam_pos}

            for index in tqdm(range(int(1.0 * len(self.imgs[self.split][self.cam_pos[0]][self.image_modes[0]])))):
                input_list, lbl_list = self.__getitem__(index)
                img_list, aux_list = input_list['rgb'], input_list['d']

                for i, p in enumerate(self.cam_pos):
                    for n in self.name2id:
                        pixel_stats[p][n].append((1. * torch.sum(lbl_list[i] == self.name2id[n]).tolist() / (list(lbl_list[i].size())[0] * list(lbl_list[i].size())[1])))

            pickle.dump(pixel_stats, open(savefile, "wb"))

        pixel_stats_summary = {p: {n: {"mean": 0} for n in self.name2id} for p in self.cam_pos}
        for i, p in enumerate(self.cam_pos):
            for n in self.name2id:
                pixel_stats_summary[p][n]["mean"] = np.mean(pixel_stats[p][n])
                pixel_stats_summary[p][n]["var"] = np.std(pixel_stats[p][n])

        # print(pixel_stats_summary)

        # for cam_pos in self.cam_pos:
        #     for image_mode in self.image_modes:
        #         for index in range(len(self.imgs[self.split][cam_pos][image_mode])):
        #             print(index)

    def tuple_to_folder_name(self, path_tuple):
        start = path_tuple[1]
        end = path_tuple[2]
        path = str(start[0]) + '_' + str(-start[1]) + '__' + str(end[0]) + '_' + str(-end[1])

        return path

    def generate_image_path(self, dataset_div):
        # (pathid, start, end, distance, region)
        # dataset_div= {'train':{'skyscraper':[0,[]],'suburban':[0,[]],'shopping':[0,[]]},
        #               'val'  :{'skyscraper':[0,[]],'suburban':[0,[]],'shopping':[0,[]]},
        #               'test' :{'skyscraper':[0,[]],'suburban':[0,[]],'shopping':[0,[]]}}

        # Merge across regions
        train_path_list = []
        recal_path_list = []
        val_path_list = []
        test_path_list = []
        for region in ['skyscraper', 'suburban', 'shopping']:
            for train_one_path in dataset_div['train'][region][1]:
                train_path_list.append(self.tuple_to_folder_name(train_one_path))

            for recal_one_path in dataset_div['recal'][region][1]:
                recal_path_list.append(self.tuple_to_folder_name(recal_one_path))

            for val_one_path in dataset_div['val'][region][1]:
                val_path_list.append(self.tuple_to_folder_name(val_one_path))

            for test_one_path in dataset_div['test'][region][1]:
                test_path_list.append(self.tuple_to_folder_name(test_one_path))

        split_subdirs = {}
        split_subdirs['train'] = ['-57_-194__20_-192']
        split_subdirs['recal'] = ['-57_-194__20_-192']
        split_subdirs['val'] = ['-57_-194__20_-192']
        split_subdirs['test'] = ['-57_-194__20_-192']

        return split_subdirs

    def divide_region_n_train_val_test(self):

        region_dict = {'skyscraper': [0, []], 'suburban': [0, []], 'shopping': [0, []]}
        test_ratio = 0.2
        val_ratio = 0.2
        recal_ratio = 0.1

        dataset_div = {'train': {'skyscraper': [0, []], 'suburban': [0, []], 'shopping': [0, []]},
                       'recal': {'skyscraper': [0, []], 'suburban': [0, []], 'shopping': [0, []]},
                       'val': {'skyscraper': [0, []], 'suburban': [0, []], 'shopping': [0, []]},
                       'test': {'skyscraper': [0, []], 'suburban': [0, []], 'shopping': [0, []]}}

        process_edges = []
        # label and computer distance
        for i, path in enumerate(self.all_edges):
            process_edges.append(label_region_n_compute_distance(i, path))

            # self.all_edges[i] = label_region_n_compute_distance(i,path)
            # (pathid, path_x, path_y, distance, region)
            region_dict[process_edges[i][4]][1].append(process_edges[i])
            region_dict[process_edges[i][4]][0] = region_dict[process_edges[i][4]][0] + process_edges[i][3]

        for region_type, distance_and_path_list in region_dict.items():
            total_distance = distance_and_path_list[0]
            recal_distance = total_distance * recal_ratio
            test_distance = total_distance * test_ratio
            val_distance = total_distance * val_ratio

            path_list = distance_and_path_list[1]
            tem_list = copy.deepcopy(path_list)

            sum_distance = 0

            # Train Set
            dataset_div['train'][region_type][0] = total_distance - sum_distance
            dataset_div['train'][region_type][1] = tem_list

        # for div_type, one_region_dict in dataset_div.items():
        # ## Train, Val, Test
        #     print(div_type)
        #     for region_type, distance_and_path_list in one_region_dict.items():
        #         print (region_type)
        #         print(distance_and_path_list[0])
        #     print('=========================================')

        color = ['red', 'yellow', 'green', 'blue']
        ## Visualiaztion with respect to region
        fig, ax = plt.subplots(figsize=(30, 15))
        div_type = 'train'

        vis_txt_height = 800
        for div_type in ['train', 'recal', 'val', 'test']:
            for region in ['skyscraper', 'suburban', 'shopping']:
                vis_path_list = dataset_div[div_type][region][1]
                for path in vis_path_list:
                    x = [path[1][0], path[2][0]]
                    y = [path[1][1], path[2][1]]

                    if region == 'skyscraper':
                        ax.plot(x, y, color='red', zorder=1, lw=3)
                    elif region == 'suburban':
                        ax.plot(x, y, color='blue', zorder=1, lw=3)
                    elif region == 'shopping':
                        ax.plot(x, y, color='green', zorder=1, lw=3)

                    ax.scatter(x, y, color='black', s=120, zorder=2)

                # Visualize distance text
                distance = dataset_div[div_type][region][0]
                if region == 'skyscraper':
                    ax.annotate(div_type + ' - ' + region + ': ' + str(distance), (-800, vis_txt_height), fontsize=20, color='red')
                elif region == 'suburban':
                    ax.annotate(div_type + ' - ' + region + ': ' + str(distance), (-800, vis_txt_height), fontsize=20, color='blue')
                elif region == 'shopping':
                    ax.annotate(div_type + ' - ' + region + ': ' + str(distance), (-800, vis_txt_height), fontsize=20, color='green')
                vis_txt_height -= 30

        plt.savefig('region.png', dpi=200)
        # plt.close()

        ## Visualization with respect to train/val/test
        fig, ax = plt.subplots(figsize=(30, 15))
        div_type = 'train'
        vis_txt_height = 800
        for div_type in ['train', 'recal', 'val', 'test']:
            for region in ['skyscraper', 'suburban', 'shopping']:
                vis_path_list = dataset_div[div_type][region][1]
                for path in vis_path_list:
                    x = [path[1][0], path[2][0]]
                    y = [path[1][1], path[2][1]]

                    if div_type == 'train':
                        ax.plot(x, y, color='red', zorder=1, lw=3)
                    elif div_type == 'recal':
                        ax.plot(x, y, color='yellow', zorder=1, lw=3)
                    elif div_type == 'val':
                        ax.plot(x, y, color='blue', zorder=1, lw=3)
                    elif div_type == 'test':
                        ax.plot(x, y, color='green', zorder=1, lw=3)

                    ax.scatter(x, y, color='black', s=120, zorder=2)

                # Visualize distance text
                distance = dataset_div[div_type][region][0]
                if div_type == 'train':
                    ax.annotate(div_type + ' - ' + region + ': ' + str(distance), (-800, vis_txt_height), fontsize=20, color='red')
                elif div_type == 'recal':
                    ax.annotate(div_type + ' - ' + region + ': ' + str(distance), (-800, vis_txt_height), fontsize=20, color='yellow')
                elif div_type == 'val':
                    ax.annotate(div_type + ' - ' + region + ': ' + str(distance), (-800, vis_txt_height), fontsize=20, color='blue')
                elif div_type == 'test':
                    ax.annotate(div_type + ' - ' + region + ': ' + str(distance), (-800, vis_txt_height), fontsize=20, color='green')
                vis_txt_height -= 30

                # ax.annotate(txt, (x, y))
        plt.savefig('train_val_test.png', dpi=200)
        # plt.close()

        return dataset_div

    def __len__(self):
        """__len__"""
        return len(self.imgs[self.split][self.cam_pos[0]][self.image_modes[0]])

    def __getitem__(self, index):
        """__getitem__

        :param index:
        """
        input_list = {'rgb': [],
                      'd': [],
                      'rgb_display': [],
                      'd_display': []}
        lbl_list = []
        start_ts = time.time()

        for camera in self.cam_pos:

            img_path, mask_path = self.imgs[self.split][camera]['scene'][index], self.imgs[self.split][camera]['segmentation'][index]
            # print(img_path)
            img, mask = np.array(cv2.imread(img_path), dtype=np.uint8)[:, :, :3], np.array(cv2.imread(mask_path), dtype=np.uint8)[:, :, :3]

            if any(['depth_encoded' == mode for mode in self.image_modes]):
                depth_path = self.imgs[self.split][camera]['depth_encoded'][index]
                depth_raw = np.array(cv2.imread(depth_path), dtype=np.uint8)
                depth = np.array((256 ** 3) * depth_raw[:, :, 0] +
                                 (256 ** 2) * depth_raw[:, :, 1] +
                                 (256 ** 1) * depth_raw[:, :, 2], dtype=np.uint32).view(np.float32)
            else:
                depth_path = self.imgs[self.split][camera]['depth'][index]
                depth = np.array(cv2.imread(depth_path), dtype=np.uint8)

            degradation = self.dgrd[self.split][camera]['scene'][index]
            if not degradation is None:
                img, depth = self.degradation(degradation, img, depth)

            lbl = self.ignore_index * np.ones((img.shape[0], img.shape[1]), dtype=np.uint8)
            for i, name in self.id2name.items():
                for color in self.name2color[name]:
                    lbl[(mask == color[::-1]).all(-1)] = i

            if self.augmentations is not None:
                img, lbl, depth = self.augmentations(img, lbl, depth)

            img, lbl, depth, img_display, depth_display = self.transform(img, lbl, depth)
            input_list['rgb'].append(img)
            input_list['d'].append(depth)
            input_list['rgb_display'].append(img_display)
            input_list['d_display'].append(depth_display)

            lbl_list.append(lbl)

        # return input_list, lbl_list
        return input_list['rgb']

    def degradation(self, degradation, img, depth):

        degradation = yaml.load((degradation))

        if degradation['type'] in key2deg.keys():
            if "rgb" in degradation['channel']:
                img = key2deg[degradation['type']](img, int(degradation['value']))
            if "d" in degradation['channel']:
                depth = key2deg[degradation['type']](depth, int(degradation['value']))
        else:
            print("Corruption Type Not Implemented")

        return img, depth

    def transform(self, img, lbl, aux):
        """transform

        :param img:
        :param lbl:
        """
        img = cv2.resize(img, (self.img_size[0], self.img_size[1]))  # uint8 with RGB mode
        aux = cv2.resize(aux, (self.img_size[0], self.img_size[1]))  # uint8 with RGB mode
        # img = img[:, :, ::-1]  # RGB -> BGR

        img = img.astype(np.float64)
        aux = aux.astype(np.float64)

        img_display = img.copy()
        aux_display = aux.copy()

        if self.img_norm:
            img = np.divide((img.astype(float) - self.mean[:3]), self.std[:3])
            aux = np.divide((aux.astype(float) - self.mean[3:]), self.std[3:])

        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)
        img_display = img_display.transpose(2, 0, 1)

        if not any(['depth_encoded' == mode for mode in self.image_modes]):
            aux = aux.transpose(2, 0, 1)
            aux_display = aux_display.transpose(2, 0, 1)

        classes = np.unique(lbl)
        lbl = lbl.astype(float)
        lbl = cv2.resize(lbl, (self.img_size[0], self.img_size[1]), interpolation=cv2.INTER_NEAREST)  # , "nearest", mode="F")
        lbl = lbl.astype(int)

        # if not np.all(classes == np.unique(lbl)):
        #     print("WARN: resizing labels yielded fewer classes")

        if not np.all(np.unique(lbl[lbl != self.ignore_index]) < self.n_classes):
            print("after det", classes, np.unique(lbl))
            raise ValueError("Segmentation map contained invalid class values")

        img = torch.from_numpy(img).float()
        aux = torch.from_numpy(aux).float()
        img_display = torch.from_numpy(img_display).float()
        aux_display = torch.from_numpy(aux_display).float()
        lbl = torch.from_numpy(lbl).long()

        return img, lbl, aux, img_display, aux_display

    def decode_segmap(self, temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for i, name in self.id2name.items():
            r[(temp == i)] = self.name2color[name][0][0]
            g[(temp == i)] = self.name2color[name][0][1]
            b[(temp == i)] = self.name2color[name][0][2]

            # r[temp == l] = self.label_colours[l][0]
            # g[temp == l] = self.label_colours[l][1]
            # b[temp == l] = self.label_colours[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb

    # def encode_segmap(self, mask):
    #     # Put all void classes to zero
    #     for _voidc in self.void_classes:
    #         mask[mask == _voidc] = self.ignore_index
    #     for _validc in self.valid_classes:
    #         mask[mask == _validc] = self.class_map[_validc]
    #     return mask
