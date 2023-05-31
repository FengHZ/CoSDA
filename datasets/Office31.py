from os import path
import os
from torch.utils.data import DataLoader, DistributedSampler
from .loader_utils import ImageList, get_regular_transforms, get_ddp_generator


class Office31(ImageList):
    """Office31 Dataset.

    Args:
        root (str): Root directory of dataset
        task (str): The task (domain) to create dataset. Choices include ``'A'``: amazon, \
            ``'D'``: dslr and ``'W'``: webcam.
        download (bool, optional): If true, downloads the dataset from the internet and puts it \
            in root directory. If dataset is already downloaded, it is not downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image and returns a \
            transformed version. E.g, :class:`torchvision.transforms.RandomCrop`.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.

    .. note:: In `root`, there will exist following files after downloading.
        ::
            amazon/
                images/
                    backpack/
                        *.jpg
                        ...
            dslr/
            webcam/
            image_list/
                amazon.txt
                dslr.txt
                webcam.txt
    """
    download_list = [
        ("image_list", "image_list.zip", "https://cloud.tsinghua.edu.cn/f/d9bca681c71249f19da2/?dl=1"),
        ("amazon", "amazon.tgz", "https://cloud.tsinghua.edu.cn/f/edc8d1bba1c740dc821c/?dl=1"),
        ("dslr", "dslr.tgz", "https://cloud.tsinghua.edu.cn/f/ca6df562b7e64850ad7f/?dl=1"),
        ("webcam", "webcam.tgz", "https://cloud.tsinghua.edu.cn/f/82b24ed2e08f4a3c8888/?dl=1"),
    ]
    image_list = {
        "amazon": "image_list/amazon.txt",
        "dslr": "image_list/dslr.txt",
        "webcam": "image_list/webcam.txt"
    }
    CLASSES = ['back_pack', 'bike', 'bike_helmet', 'bookcase', 'bottle', 'calculator', 'desk_chair', 'desk_lamp',
               'desktop_computer', 'file_cabinet', 'headphones', 'keyboard', 'laptop_computer', 'letter_tray',
               'mobile_phone', 'monitor', 'mouse', 'mug', 'paper_notebook', 'pen', 'phone', 'printer', 'projector',
               'punchers', 'ring_binder', 'ruler', 'scissors', 'speaker', 'stapler', 'tape_dispenser', 'trash_can']

    def __init__(self, dataset_path: str, domain_name: str, transform=None, label_transform=None):
        assert domain_name in self.image_list
        data_list_file = os.path.join(dataset_path, self.image_list[domain_name])
        super(Office31, self).__init__(dataset_path, Office31.CLASSES, data_list_file=data_list_file,
                                       transform=transform,
                                       label_transform=label_transform)

    def __getitem__(self, index: int):
        origin = super().__getitem__(index)
        return origin[0], origin[1], index

    @classmethod
    def domains(cls):
        return list(cls.image_list.keys())


def get_office31_dloader(base_path, domain_name, batch_size, num_workers):
    dataset_path = path.join(base_path, 'dataset', 'Office31')
    transforms_train, transforms_test = get_regular_transforms()
    g = get_ddp_generator()
    train_dataset = Office31(dataset_path, domain_name, transform=transforms_train)
    train_sampler = DistributedSampler(train_dataset)
    test_dataset = Office31(dataset_path, domain_name, transform=transforms_test)
    test_sampler = DistributedSampler(test_dataset, shuffle=False)
    train_dloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                               shuffle=False, sampler=train_sampler, generator=g, persistent_workers=True)
    test_dloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                              shuffle=False, sampler=test_sampler)
    return train_dloader, test_dloader
