import os

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from utils.joint_transforms import Compose, JointResize, RandomHorizontallyFlip, RandomRotate
from utils.misc import construct_print

mean_rgb = np.array([0.447, 0.407, 0.386])
std_rgb = np.array([0.244, 0.250, 0.253])

def _get_ext(path_list):
    ext_list = list(set([os.path.splitext(p)[1] for p in path_list]))
    if len(ext_list) != 1:
        if '.png' in ext_list:
            ext = '.png'
        elif '.jpg' in ext_list:
            ext = '.jpg'
        elif '.bmp' in ext_list:
            ext = '.bmp'
        else:
            raise NotImplementedError
        construct_print(f"数据文件夹中包含多种扩展名，这里仅使用{ext}")
    elif len(ext_list) == 1:
        ext = ext_list[0]
    else:
        raise NotImplementedError
    return ext

def _make_unlabeled_dataset(root, split):
    img_path = os.path.join(root, split + '_images')
    depth_path = os.path.join(root, split + '_depth')
    
    img_list = os.listdir(img_path)
    depth_list = os.listdir(depth_path)
    
    img_ext = _get_ext(img_list)
    depth_ext = _get_ext(depth_list)
    
    # relative path to main file name
    img_list = [os.path.splitext(f)[0] for f in img_list if f.endswith(img_ext)]
    return [ (os.path.join(img_path, img_name + img_ext), \
              os.path.join(depth_path, img_name + depth_ext)) for img_name in img_list ]


def _make_dataset(root, split):
    img_path = os.path.join(root, split + '_images')
    depth_path = os.path.join(root, split + '_depth')
    mask_path = os.path.join(root, split + '_masks')
    
    img_list = os.listdir(img_path)
    depth_list = os.listdir(depth_path)
    mask_list = os.listdir(mask_path)
    
    img_ext = _get_ext(img_list)
    depth_ext = _get_ext(depth_list)
    mask_ext = _get_ext(mask_list)
    
    img_list = [os.path.splitext(f)[0] for f in mask_list if f.endswith(mask_ext)]
    return [(os.path.join(img_path, img_name + img_ext),
             os.path.join(depth_path, img_name + depth_ext),
             os.path.join(mask_path, img_name + mask_ext),
             )
            for img_name in img_list]


def _read_list_from_file(list_filepath):
    img_list = []
    with open(list_filepath, mode='r', encoding='utf-8') as openedfile:
        line = openedfile.readline()
        while line:
            img_list.append(line.split()[0])
            line = openedfile.readline()
    return img_list


def _make_test_dataset_from_list(list_filepath, prefix=('.jpg', '.png')):
    img_list = _read_list_from_file(list_filepath)
    return [(os.path.join(os.path.join(os.path.dirname(img_path), 'test_images'),
                          os.path.basename(img_path) + prefix[0]),
             os.path.join(os.path.join(os.path.dirname(img_path), 'test_masks'),
                          os.path.basename(img_path) + prefix[1]))
            for img_path in img_list]


class TestImageFolder(Dataset):
    def __init__(self, root, in_size, prefix):
        if os.path.isdir(root):
            construct_print(f"{root} is an image folder, we will test on it.")
            self.imgs = _make_dataset(root, split = 'test')
        elif os.path.isfile(root):
            construct_print(f"{root} is a list of images, we will use these paths to read the "
                            f"corresponding image")
            self.imgs = _make_test_dataset_from_list(root, prefix=prefix)
        else:
            raise NotImplementedError
        self.test_img_trainsform = transforms.Compose([
            # 输入的如果是一个tuple，则按照数据缩放，但是如果是一个数字，则按比例缩放到短边等于该值
            transforms.Resize((in_size, in_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.test_depth_trainsform = transforms.Compose([
            transforms.Resize((in_size, in_size)),
            transforms.ToTensor()
        ])
    def __getitem__(self, index):
        img_path, depth_path, mask_path = self.imgs[index]
        
        img = Image.open(img_path).convert('RGB')
        depth = Image.open(depth_path).convert('L')
        img_name = (img_path.split(os.sep)[-1]).split('.')[0]
        
        img = self.test_img_trainsform(img).float()
        depth = self.test_depth_trainsform(depth).float()
        # depth = (depth - torch.min(depth)) / (torch.max(depth) - torch.min(depth))
        return img, depth, mask_path, img_name
    
    def __len__(self):
        return len(self.imgs)


class TestUnlabeledImageFolder(Dataset):
    def __init__(self, root, in_size, prefix):
        if os.path.isdir(root):
            construct_print(f"{root} is an image folder, we will test on it.")
            self.imgs = _make_unlabeled_dataset(root, split = 'test')
        elif os.path.isfile(root):
            raise NotImplementedError
        else:
            raise NotImplementedError
        self.test_img_trainsform = transforms.Compose([
            transforms.Resize((in_size, in_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.test_depth_trainsform = transforms.Compose([
            transforms.Resize((in_size, in_size)),
            transforms.ToTensor()
        ])
    def __getitem__(self, index):
        img_path, depth_path= self.imgs[index]
        
        img = Image.open(img_path).convert('RGB')
        depth = Image.open(depth_path).convert('L')
        img_name = (img_path.split(os.sep)[-1]).split('.')[0]
        
        img = self.test_img_trainsform(img).float()
        depth = self.test_depth_trainsform(depth).float()
        depth = (depth - torch.min(depth)) / (torch.max(depth) - torch.min(depth) + torch.tensor(1.0e-6))
        return img, depth, img_name
    
    def __len__(self):
        return len(self.imgs)


def _make_train_dataset_from_list(list_filepath, prefix=('.jpg', '.png')):
    # list_filepath = '/home/lart/Datasets/RGBDSaliency/FinalSet/rgbd_train_jw.lst'
    img_list = _read_list_from_file(list_filepath)
    return [(os.path.join(os.path.join(os.path.dirname(img_path), 'train_images'),
                          os.path.basename(img_path) + prefix[0]),
             os.path.join(os.path.join(os.path.dirname(img_path), 'train_masks'),
                          os.path.basename(img_path) + prefix[1]))
            for img_path in img_list]


class TrainImageFolder(Dataset):
    def __init__(self, root, in_size, prefix, use_bigt=False):
        self.use_bigt = use_bigt
        if os.path.isdir(root):
            construct_print(f"{root} is an image folder, we will train on it.")
            self.imgs = _make_dataset(root, split = 'train')
        elif os.path.isfile(root):
            construct_print(f"{root} is a list of images, we will use these paths to read the "
                            f"corresponding image")
            self.imgs = _make_train_dataset_from_list(root, prefix=prefix)
        else:
            raise NotImplementedError

        self.train_joint_transform = Compose([
            JointResize(in_size),
            RandomHorizontallyFlip(),
            RandomRotate(10)
        ])
        self.train_img_transform = transforms.Compose([
            transforms.ColorJitter(0.1, 0.1, 0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])  # 处理的是Tensor
        ])
        self.train_depth_transform = transforms.ToTensor()
        self.train_mask_transform = transforms.ToTensor()
    
    def __getitem__(self, index):
        img_path, depth_path, mask_path = self.imgs[index]
        
        img = Image.open(img_path)
        depth = Image.open(depth_path)
        mask = Image.open(mask_path)
        if len(img.split()) != 3:
            img = img.convert('RGB')
        if len(depth.split()) == 3:
            depth = depth.convert('L')
        if len(mask.split()) == 3:
            mask = mask.convert('L')

        img, depth, mask = self.train_joint_transform(img, depth, mask)
        mask = self.train_mask_transform(mask).float()
        img = self.train_img_transform(img).float()
        depth = self.train_depth_transform(depth).float()
        # depth = (depth - torch.min(depth)) / (torch.max(depth) - torch.min(depth))
        if self.use_bigt:
            mask = mask.ge(0.5).float()  # 二值化
        
        img_name = (img_path.split(os.sep)[-1]).split('.')[0]
        
        return img, depth, mask, img_name

    def __len__(self):
        return len(self.imgs)

class TrainMTImageFolder(Dataset):
    def __init__(self, root, unlabeled_root, in_size, prefix, use_bigt=False):
        self.in_size = in_size
        self.use_bigt = use_bigt
        if os.path.isdir(root):
            construct_print(f"{root} is an image folder, we will train on it.")
            self.imgs = _make_dataset(root, split = 'train')
        elif os.path.isfile(root):
            construct_print(f"{root} is a list of images, we will use these paths to read the "
                            f"corresponding image")
            self.imgs = _make_train_dataset_from_list(root, prefix=prefix)
        else:
            raise NotImplementedError
        if os.path.isdir(unlabeled_root):
            construct_print(f"{unlabeled_root} is an image folder, we will conduct MT on it.")
            self.unlabeled_imgs = _make_unlabeled_dataset(unlabeled_root, split = 'train')
        elif os.path.isfile(unlabeled_root):
            raise NotImplementedError
        else:
            raise NotImplementedError

        self.train_joint_transform = Compose([
            JointResize(in_size),
            RandomHorizontallyFlip(),
            RandomRotate(10)
        ])
        self.train_img_transform = transforms.Compose([
            transforms.ColorJitter(0.1, 0.1, 0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])  # 处理的是Tensor
        ])
        self.train_depth_transform = transforms.ToTensor()
        self.train_mask_transform = transforms.ToTensor()
    
    def __getitem__(self, index):
        if index < len(self.imgs):
            img_path, depth_path, mask_path = self.imgs[index]
            
            img = Image.open(img_path)
            depth = Image.open(depth_path)
            mask = Image.open(mask_path)
            if len(img.split()) != 3:
                img = img.convert('RGB')
            if len(depth.split()) == 3:
                depth = depth.convert('L')
            if len(mask.split()) == 3:
                mask = mask.convert('L')

            img, depth, mask = self.train_joint_transform(img, depth, mask)
            mask = self.train_mask_transform(mask).float()
            img = self.train_img_transform(img).float()
            depth = self.train_depth_transform(depth).float()
            # depth = (depth - torch.min(depth)) / (torch.max(depth) - torch.min(depth))
            if self.use_bigt:
                mask = mask.ge(0.5).float()  # 二值化
            
            img_name = (img_path.split(os.sep)[-1]).split('.')[0]
            
            return img, depth, mask, img_name
        else:
            index -= len(self.imgs)
            # without unlabeled_mask_path
            unlabeled_img_path, unlabeled_depth_path = self.unlabeled_imgs[index]
            unlabeled_img = Image.open(unlabeled_img_path)
            unlabeled_depth = Image.open(unlabeled_depth_path)

            if len(unlabeled_img.split()) != 3:
                unlabeled_img = unlabeled_img.convert('RGB')
            if len(unlabeled_depth.split()) == 3:
                unlabeled_depth = unlabeled_depth.convert('L')            
            
            unlabeled_img, unlabeled_depth = self.train_joint_transform(unlabeled_img, unlabeled_depth)
            unlabeled_img = self.train_img_transform(unlabeled_img).float()
            unlabeled_depth = self.train_depth_transform(unlabeled_depth).float()
            unlabeled_depth = (unlabeled_depth - torch.min(unlabeled_depth)) / (torch.max(unlabeled_depth) - torch.min(unlabeled_depth) + torch.tensor(1.0e-6))
            unlabeled_mask = torch.zeros((1, self.in_size, self.in_size)).float()
            
            unlabeled_img_name = (unlabeled_img_path.split(os.sep)[-1]).split('.')[0]
            
            return unlabeled_img, unlabeled_depth, unlabeled_mask, unlabeled_img_name             
    
    def __len__(self):
        return len(self.imgs) + len(self.unlabeled_imgs)

    def get_primary_secondary_indices(self):
        return np.arange(len(self.imgs)), np.arange(len(self.imgs), len(self.unlabeled_imgs))

class TrainSSImageFolder(Dataset):
    def __init__(self, root, unlabeled_root, in_size, prefix, is_labeled_rotation,
                 use_bigt=False, rotations = (0, 90, 180, 270)):
        self.in_size = in_size
        self.is_labeled_rotation = is_labeled_rotation
        self.use_bigt = use_bigt
        self.rotations = rotations
        self.times = len(rotations)

        if os.path.isdir(root):
            construct_print(f"{root} is an image folder, we will train on it.")
            self.imgs = _make_dataset(root, split = 'train')
        elif os.path.isfile(root):
            construct_print(f"{root} is a list of images, we will use these paths to read the "
                            f"corresponding image")
            self.imgs = _make_train_dataset_from_list(root, prefix=prefix)
        else:
            raise NotImplementedError
        if os.path.isdir(unlabeled_root):
            construct_print(f"{unlabeled_root} is an image folder, we will conduct MT on it.")
            self.unlabeled_imgs = _make_unlabeled_dataset(unlabeled_root, split = 'train')
        elif os.path.isfile(unlabeled_root):
            raise NotImplementedError
        else:
            raise NotImplementedError

        self.train_joint_transform = JointResize(in_size)

        self.train_img_transform = transforms.Compose([
            transforms.ColorJitter(0.1, 0.1, 0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])  # 处理的是Tensor
        ])
        self.train_depth_transform = transforms.ToTensor()
        self.train_mask_transform = transforms.ToTensor()
    
    def __getitem__(self, index):
        # main_index, rotate_index = index // self.times, index % self.times
        main_index = index
        rotate_index = np.random.randint(self.times)
        if main_index < len(self.imgs):
            img_path, depth_path, mask_path = self.imgs[main_index]
            
            img = Image.open(img_path)
            depth = Image.open(depth_path)
            mask = Image.open(mask_path)
            if len(img.split()) != 3:
                img = img.convert('RGB')
            if len(depth.split()) == 3:
                depth = depth.convert('L')
            if len(mask.split()) == 3:
                mask = mask.convert('L')

            img, depth, mask = self.train_joint_transform(img, depth, mask)
            if self.is_labeled_rotation:
                img = img.rotate(self.rotations[rotate_index])
                depth = depth.rotate(self.rotations[rotate_index])
                mask = mask.rotate(self.rotations[rotate_index])

            mask = self.train_mask_transform(mask).float()
            img = self.train_img_transform(img).float()
            depth = self.train_depth_transform(depth).float()
            # depth = (depth - torch.min(depth)) / (torch.max(depth) - torch.min(depth))
            if self.use_bigt:
                mask = mask.ge(0.5).float()  # 二值化
            
            # rotate_label = torch.zeros((self.times), dtype = torch.int64).scatter_(0, torch.LongTensor([ rotate_index ]), torch.LongTensor([ 1 ])).long()
            rotate_label = torch.tensor(rotate_index).long()

            img_name = (img_path.split(os.sep)[-1]).split('.')[0]
            
            return img, depth, mask, rotate_label, img_name
        else:
            main_index -= len(self.imgs)
            # without unlabeled_mask_path
            unlabeled_img_path, unlabeled_depth_path = self.unlabeled_imgs[main_index]
            unlabeled_img = Image.open(unlabeled_img_path)
            unlabeled_depth = Image.open(unlabeled_depth_path)

            if len(unlabeled_img.split()) != 3:
                unlabeled_img = unlabeled_img.convert('RGB')
            if len(unlabeled_depth.split()) == 3:
                unlabeled_depth = unlabeled_depth.convert('L')            
            
            unlabeled_img, unlabeled_depth = self.train_joint_transform(unlabeled_img, unlabeled_depth)
            unlabeled_img = unlabeled_img.rotate(self.rotations[rotate_index])
            unlabeled_depth = unlabeled_depth.rotate(self.rotations[rotate_index])

            unlabeled_img = self.train_img_transform(unlabeled_img).float()
            unlabeled_depth = self.train_depth_transform(unlabeled_depth).float()
            unlabeled_depth = (unlabeled_depth - torch.min(unlabeled_depth)) / (torch.max(unlabeled_depth) - torch.min(unlabeled_depth) + torch.tensor(1.0e-6))
            
            unlabeled_mask = torch.zeros((1, self.in_size, self.in_size)).float() # dummy

            # unlabeled_rotate_label = torch.zeros((self.times), dtype = torch.int64).scatter_(0, torch.LongTensor([ rotate_index ]), torch.LongTensor([ 1 ]))
            unlabeled_rotate_label = torch.tensor(rotate_index).long()

            unlabeled_img_name = (unlabeled_img_path.split(os.sep)[-1]).split('.')[0]
            
            return unlabeled_img, unlabeled_depth, unlabeled_mask, unlabeled_rotate_label, unlabeled_img_name             
    
    def __len__(self):
        return (len(self.imgs) + len(self.unlabeled_imgs))
        # return (len(self.imgs) + len(self.unlabeled_imgs)) * self.times

    def get_primary_secondary_indices(self):
        return np.arange(len(self.imgs)), np.arange(len(self.imgs), len(self.unlabeled_imgs))
        # return np.arange(len(self.imgs) * self.times), np.arange(len(self.imgs) * self.times, len(self.unlabeled_imgs) * self.times)


if __name__ == '__main__':
    img_list = _make_train_dataset_from_list()
    construct_print(len(img_list))
