from os import path
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from datasets.Visda2017 import VisDA2017
from train.dac.autoaugment import ImageNetPolicy


def get_tailored_transforms_train(resize_size=256, crop_size=224):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform_weak = transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), normalize
    ])

    transform_strong = transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        ImageNetPolicy(),
        transforms.ToTensor(), normalize
    ])

    return transform_weak, transform_strong


def get_tailored_transforms_test():
    transforms_test = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))
    ])

    return transforms_test


class Visda2017DaC(VisDA2017):

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        img = self.loader(path)
        transforms_weak, transorms_strong = self.transform
        img_weak = transforms_weak(img)
        img_strong_1 = transorms_strong(img)
        img_strong_2 = transorms_strong(img)

        return img_weak, label, img_strong_1, img_strong_2, index


def get_visda17_dac_dloader(base_path, domain_name, batch_size, num_workers):
    dataset_path = path.join(base_path, 'dataset', 'Visda2017')
    train_dataset = Visda2017DaC(dataset_path,
                                 domain_name,
                                 transform=get_tailored_transforms_train())

    test_dataset = VisDA2017(dataset_path,
                             domain_name,
                             transform=get_tailored_transforms_test())
    train_dloader = DataLoader(train_dataset,
                               batch_size=batch_size,
                               num_workers=num_workers,
                               pin_memory=True,
                               shuffle=True)
    test_dloader = DataLoader(test_dataset,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              pin_memory=True,
                              shuffle=False)
    return train_dloader, test_dloader
