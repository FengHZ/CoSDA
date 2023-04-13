from .loader_utils import ImageList
from os import path
import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


class VisDA2017(ImageList):
    """`VisDA-2017 <http://ai.bu.edu/visda-2017/assets/attachments/VisDA_2017.pdf>`_ Dataset

     Args:
         dataset_path (str): Root directory of dataset
         domain_name (str): The task (domain) to create dataset. Choices include ``'Synthetic'``: synthetic images and \
             ``'Real'``: real-world images.
         download (bool, optional): If true, downloads the dataset from the internet and puts it \
             in root directory. If dataset is already downloaded, it is not downloaded again.
         transform (callable, optional): A function/transform that  takes in an PIL image and returns a \
             transformed version. E.g, ``transforms.RandomCrop``.
         target_transform (callable, optional): A function/transform that takes in the target and transforms it.

     .. note:: In `root`, there will exist following files after downloading.
         ::
             train/
                 aeroplance/
                     *.png
                     ...
             validation/
             image_list/
                 train.txt
                 validation.txt
     """
    image_list = {
        "Synthetic": "image_list/train.txt",
        "Real": "image_list/validation.txt"
    }
    CLASSES = ['aeroplane', 'bicycle', 'bus', 'car', 'horse', 'knife',
               'motorcycle', 'person', 'plant', 'skateboard', 'train', 'truck']

    def __init__(self, dataset_path: str, domain_name: str, transform=None, label_transform=None):
        assert domain_name in self.image_list
        data_list_file = os.path.join(dataset_path, self.image_list[domain_name])
        super(VisDA2017, self).__init__(dataset_path, VisDA2017.CLASSES, data_list_file=data_list_file,
                                        transform=transform,label_transform=label_transform)

    def __getitem__(self, index: int):
        origin = super().__getitem__(index)
        return origin[0], origin[1], index

    @classmethod
    def domains(cls):
        return list(cls.image_list.keys())


def get_visda17_dloader(base_path, domain_name, batch_size, num_workers):
    # domain name in ["Synthetic","Real"]
    dataset_path = path.join(base_path, 'dataset', 'Visda2017')
    transforms_train = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.75, 1)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    transforms_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    train_dataset = VisDA2017(dataset_path, domain_name, transform=transforms_train)
    test_dataset = VisDA2017(dataset_path, domain_name, transform=transforms_test)
    train_dloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                               shuffle=True)
    test_dloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                              shuffle=False)
    return train_dloader, test_dloader
