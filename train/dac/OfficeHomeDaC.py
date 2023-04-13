from os import path
from typing import Tuple, Any
from torch.utils.data import DataLoader

from datasets.OfficeHome import OfficeHome
from train.dac.Visda2017DaC import get_tailored_transforms_train, get_tailored_transforms_test


class OfficeHomeDac(OfficeHome):

    def __getitem__(self, index: int) -> Tuple[Any, int, Any, Any, int]:
        path, label = self.samples[index]
        img = self.loader(path)
        transforms_weak, transorms_strong = self.transform
        img_weak = transforms_weak(img)
        img_strong_1 = transorms_strong(img)
        img_strong_2 = transorms_strong(img)

        return img_weak, label, img_strong_1, img_strong_2, index


def get_officehome_dac_dloader(base_path, domain_name, batch_size,
                               num_workers):
    dataset_path = path.join(base_path, 'dataset', 'OfficeHome')
    train_dataset = OfficeHomeDac(dataset_path,
                                  domain_name,
                                  transform=get_tailored_transforms_train())
    test_dataset = OfficeHome(dataset_path,
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