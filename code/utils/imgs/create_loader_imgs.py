from prefetch_generator import BackgroundGenerator
from torch.utils.data import DataLoader

from utils.config import arg_config
from utils.imgs.create_rgb_datasets_imgs import TestImageFolder, TrainImageFolder
from utils.misc import construct_print


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super(DataLoaderX, self).__iter__())


def _make_loader(dataset, shuffle=True, drop_last=False):
    return DataLoaderX(dataset=dataset,
                       batch_size=arg_config["batch_size"],
                       num_workers=arg_config["num_workers"],
                       shuffle=shuffle, drop_last=drop_last,
                       pin_memory=True)


def create_loader(data_path, mode, get_length=False, prefix=('.jpg', '.png')):
    length_of_dataset = 0
    
    if mode == 'train':
        construct_print(f"Training on: {data_path}")
        train_set = TrainImageFolder(data_path,
                                     in_size=arg_config["input_size"],
                                     prefix=prefix,
                                     use_bigt=arg_config['use_bigt'])
        loader = _make_loader(train_set, shuffle=True, drop_last=True)
        length_of_dataset = len(train_set)
    elif mode == 'test':
        if data_path is not None:
            construct_print(f"Testing on: {data_path}")
            test_set = TestImageFolder(data_path,
                                       in_size=arg_config["input_size"],
                                       prefix=prefix)
            loader = _make_loader(test_set, shuffle=False, drop_last=False)
            length_of_dataset = len(test_set)
        else:
            construct_print(f"No test...")
            loader = None
    else:
        raise NotImplementedError
    
    if get_length:
        return loader, length_of_dataset
    else:
        return loader
