import os
from typing import Optional, Callable, Tuple, Any, List
import torchvision.datasets as datasets
from torchvision.datasets.folder import default_loader
import torch
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np


class ImageList(datasets.VisionDataset):
    """A generic Dataset class for image classification

    Args:
        root (str): Root directory of dataset
        classes (list[str]): The names of all the classes
        data_list_file (str): File to read the image list from.
        transform (callable, optional): A function/transform that  takes in an PIL image \
            and returns a transformed version. E.g, :class:`torchvision.transforms.RandomCrop`.
        label_transform (callable, optional): A function/transform that takes in the target and transforms it.

    .. note:: In `data_list_file`, each line has 2 values in the following format.
        ::
            source_dir/dog_xxx.png 0
            source_dir/cat_123.png 1
            target_dir/dog_xxy.png 0
            target_dir/cat_nsdf3.png 1

        The first value is the relative path of an image, and the second value is the label of the corresponding image.
        If your data_list_file has different formats, please over-ride :meth:`~ImageList.parse_data_file`.
    """

    def __init__(self, root: str, classes: List[str], data_list_file: str,
                 transform: Optional[Callable] = None, label_transform: Optional[Callable] = None):
        super().__init__(root, transform=transform, target_transform=label_transform)
        self.samples = self.parse_data_file(data_list_file)
        self.classes = classes
        self.class_to_idx = {cls: idx
                             for idx, cls in enumerate(self.classes)}
        self.loader = default_loader
        self.data_list_file = data_list_file
        self.labels = [l for (_, l) in self.samples]

    def __getitem__(self, index: int) -> Tuple[Any, int]:
        """
        Args:
            index (int): Index
            return (tuple): (image, label) where label is index of the label class.
        """
        path, label = self.samples[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None and label is not None:
            label = self.target_transform(label)
        return img, label

    def __len__(self) -> int:
        return len(self.samples)

    def parse_data_file(self, file_name: str) -> List[Tuple[str, int]]:
        """Parse file to data list

        Args:
            file_name (str): The path of data file
            return (list): List of (image path, class_index) tuples
        """
        with open(file_name, "r") as f:
            data_list = []
            for line in f.readlines():
                split_line = line.split()
                target = split_line[-1]
                path = ' '.join(split_line[:-1])
                if not os.path.isabs(path):
                    path = os.path.join(self.root, path)
                target = int(target)
                data_list.append((path, target))
        return data_list

    @property
    def num_classes(self) -> int:
        """Number of classes"""
        return len(self.classes)


def get_split_sampler(labels, test_ratio=0.1, num_classes=31):
    """
    :param labels: torch.array(long tensor)
    :param test_ratio: the ratio to split part of the data for test
    :param num_classes: 31
    :return: sampler_train,sampler_test
    """
    sampler_test = []
    sampler_train = []
    for i in range(num_classes):
        loc = torch.nonzero(labels == i)
        loc = loc.view(-1)
        # do random perm to make sure uniform sample
        test_num = round(loc.size(0) * test_ratio)
        loc = loc[torch.randperm(loc.size(0))]
        sampler_test.extend(loc[:test_num].tolist())
        sampler_train.extend(loc[test_num:].tolist())
    sampler_test = SubsetRandomSampler(sampler_test)
    sampler_train = SubsetRandomSampler(sampler_train)
    return sampler_train, sampler_test
