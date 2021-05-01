import os
import torch
from torchvision import transforms
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from lxml import etree
import numpy as np


def read_labels_dict(data_path):
    train_path = os.path.join(data_path, 'ILSVRC', 'Data', 'CLS-LOC', 'train')
    labels_dict = {}
    index = 0
    for class_name in os.listdir(train_path):
        labels_dict[class_name] = index
        index += 1
    return labels_dict


def read_train_data(data_path, labels_dict):
    file_path = os.path.join(data_path, 'ILSVRC', 'ImageSets', 'CLS-LOC', 'train_cls.txt')
    with open(os.path.join(file_path), 'r') as fid:
        lines = fid.readlines()

    train_path = os.path.join(data_path, 'ILSVRC', 'Data', 'CLS-LOC', 'train')
    paths_and_labels = []
    for line in lines:
        rel_path = line.split()[0]
        img_path = os.path.join(train_path, rel_path + '.JPEG')
        class_name = os.path.basename(rel_path).split('_')[0]
        label = labels_dict[class_name]
        paths_and_labels.append([img_path, label])

    return paths_and_labels


def parse_annotation(annotation_path, labels_dict):
    tree = etree.parse(annotation_path)
    annotation = tree.getroot()
    object = annotation.find('object')
    class_name = object.find('name').text
    class_id = labels_dict[class_name]
    return class_id


def read_val_data(data_path, labels_dict):
    file_path = os.path.join(data_path, 'ILSVRC', 'ImageSets', 'CLS-LOC', 'val.txt')
    with open(os.path.join(file_path), 'r') as fid:
        lines = fid.readlines()

    images_path = os.path.join(data_path, 'ILSVRC', 'Data', 'CLS-LOC', 'val')
    annotations_path = os.path.join(data_path, 'ILSVRC', 'Annotations', 'CLS-LOC', 'val')

    paths_and_labels = []
    for line in lines:
        img_name = line.split()[0]
        img_path = os.path.join(images_path, img_name + '.JPEG')
        ann_path = os.path.join(annotations_path, img_name + '.xml')
        label = parse_annotation(ann_path, labels_dict)
        paths_and_labels.append([img_path, label])

    return paths_and_labels


class ImageNetDataset(Dataset):
    def __init__(self, data_path, split, transform):
        self.data_path = data_path
        self.labels_dict = read_labels_dict(data_path)
        if split == 'train':
            self.paths_and_labels = read_train_data(data_path, self.labels_dict)
        else:
            self.paths_and_labels = read_val_data(data_path, self.labels_dict)
        self.transform = transform

    def __len__(self):
        return len(self.paths_and_labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.paths_and_labels[idx][0]
        image = io.imread(img_path)

        # Some image are in grayscale, make sure they have three channels:
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
            image = np.tile(image, [1, 1, 3])
        elif len(image.shape) == 3 and image.shape[2] == 4:
            # The fourth channel is probably the alpha, just ignore it.
            image = image[:, :, :3]
        assert len(image.shape) == 3
        assert image.shape[2] == 3

        label = self.paths_and_labels[idx][1]
        sample = (image, label)
        if self.transform:
            sample = self.transform(sample)
        return sample


class PadToSquare:
    def __call__(self, sample):
        image, label = sample
        height, width = image.shape[:2]

        max_side = max(width, height)
        left = int((max_side - width) / 2)
        right = max_side - width - left
        top = int((max_side - height) / 2)
        bottom = max_side - height - top

        image = np.pad(image, ((top, bottom), (left, right), (0, 0)))

        return (image, label)


class Rescale:
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample
        image = transform.resize(image, (self.output_size, self.output_size))
        # resize outputs an image with dtype float64; we want float32.
        image = image.astype(np.float32)
        return (image, label)


class SubtractMean:
    def __init__(self):
        self.image_means = np.array([123.0, 117.0, 104.0]).astype(np.float32)
        self.image_means /= 255.0
        self.image_means = np.reshape(self.image_means, [1, 1, 3])
        # self.image_means = np.tile(self.image_means, [224, 224, 1])

    def __call__(self, sample):
        image, label = sample
        image = image - self.image_means
        return (image, label)


class ToTensor:
    def __call__(self, sample):
        image, label = sample
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        sample = (torch.from_numpy(image), label)
        return sample


def get_transforms(output_size):
    transform = transforms.Compose([PadToSquare(), Rescale(output_size), SubtractMean(), ToTensor()])
    return transform