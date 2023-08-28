from PIL import Image
from os import path
from torch.utils.data import DataLoader

from datasets.DomainNet import DomainNet, read_domainnet_data
from train.dac.Visda2017DaC import get_tailored_transforms_train, get_tailored_transforms_test


class DomainNetDaC(DomainNet):

    def __getitem__(self, index):
        img = Image.open(self.data_paths[index])
        if not img.mode == "RGB":
            img = img.convert("RGB")
        label = self.data_labels[index]
        transform_weak, transform_strong = self.transforms
        img_weak = transform_weak(img)
        img_strong_1 = transform_strong(img)
        img_strong_2 = transform_strong(img)

        return img_weak, label, img_strong_1, img_strong_2, index


def get_domainnet_dac_dloader(base_path, domain_name, batch_size, num_workers):
    dataset_path = path.join(base_path, 'dataset', 'DomainNet')
    train_data_paths, train_data_labels = read_domainnet_data(dataset_path,
                                                              domain_name,
                                                              is_mini=False,
                                                              split="train")
    test_data_paths, test_data_labels = read_domainnet_data(dataset_path,
                                                            domain_name,
                                                            is_mini=False,
                                                            split="test")
    train_dataset = DomainNetDaC(train_data_paths, train_data_labels,
                                 get_tailored_transforms_train(), domain_name)
    test_dataset = DomainNet(test_data_paths, test_data_labels,
                             get_tailored_transforms_test(), domain_name)
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

def get_domainnet_mini_dac_dloader(base_path, domain_name, batch_size, num_workers):
    dataset_path = path.join(base_path, 'dataset', 'DomainNet')
    train_data_paths, train_data_labels = read_domainnet_data(dataset_path,
                                                              domain_name,
                                                              is_mini=True,
                                                              split="train")
    test_data_paths, test_data_labels = read_domainnet_data(dataset_path,
                                                            domain_name,
                                                            is_mini=True,
                                                            split="test")
    train_dataset = DomainNetDaC(train_data_paths, train_data_labels,
                                 get_tailored_transforms_train(), domain_name)
    test_dataset = DomainNet(test_data_paths, test_data_labels,
                             get_tailored_transforms_test(), domain_name)
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
