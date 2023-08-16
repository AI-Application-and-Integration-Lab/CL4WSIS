import os
import numpy as np
import torch.utils.data as data
from PIL import Image
from tqdm import tqdm
import shutil

class VOCSegmentation(data.Dataset):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Segmentation Dataset.
    Args:
        root (string): Root directory of the VOC Dataset.
        image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val``
        is_aug (bool, optional): If you want to use the augmented train set or not (default is True)
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    """

    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 indices=None,
                 as_coco=False,
                 saliency=False,
                 pseudo=None):

        self.root = os.path.expanduser(root)
        self.year = "2012"

        self.transform = transform

        self.image_set = 'train' if train else 'val'
        base_dir = "voc"
        voc_root = os.path.join(self.root, base_dir)
        splits_dir = os.path.join(voc_root, 'splits')

        if not os.path.isdir(voc_root):
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' Download it')

        mask_dir = os.path.join(voc_root, 'SegmentationClassAug')
        assert os.path.exists(mask_dir), "SegmentationClassAug not found"

        if as_coco:
            if train:
                split_f = os.path.join(splits_dir, 'train_aug_ascoco.txt')
            else:
                split_f = os.path.join(splits_dir, 'val_ascoco.txt')
        else:
            if train:
                split_f = os.path.join(splits_dir, 'train_aug.txt')
            else:
                split_f = os.path.join(splits_dir, 'val.txt')

        if not os.path.exists(split_f):
            raise ValueError(
                'Wrong image_set entered! Please use image_set="train" '
                'or image_set="trainval" or image_set="val"')

        # remove leading \n
        with open(os.path.join(split_f), "r") as f:
            file_names = [x[:-1].split(' ') for x in f.readlines()]

        # REMOVE FIRST SLASH OTHERWISE THE JOIN WILL start from root
        self.images = [(os.path.join(voc_root, x[0][1:]), os.path.join(voc_root, x[1][1:])) for x in file_names]
        
        if saliency:
            self.saliency_images = [x[0].replace("JPEGImages", "SALImages")[:-3] + "png" for x in self.images]
        else:
            self.saliency_images = None

        if pseudo is not None and train:
            if not as_coco:
                self.images = [(x[0], x[1].replace("SegmentationClassAug", f"PseudoLabels/{pseudo}/rw/")) for x in self.images]
            else:
                self.images = [(x[0], x[1].replace("SegmentationClassAugAsCoco", f"PseudoLabels/{pseudo}/rw")) for x in
                               self.images]
        if as_coco:
            self.img_lvl_labels = np.load(os.path.join(voc_root, f"cocovoc_1h_labels_{self.image_set}.npy"))
        else:
            self.img_lvl_labels = np.load(os.path.join(voc_root, f"voc_1h_labels_{self.image_set}.npy"))

        self.indices = indices if indices is not None else np.arange(len(self.images))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[self.indices[index]][0]).convert('RGB')
        target = Image.open(self.images[self.indices[index]][1])
        img_lvl_lbls = self.img_lvl_labels[self.indices[index]]

        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target, img_lvl_lbls

    def __len__(self):
        return len(self.indices)

coco_map = {
    0: 0,
    1: 5,
    2: 2,
    3: 16,
    4: 9,
    5: 44,
    6: 6,
    7: 3,
    8: 17,
    9: 62,
    10: 21,
    11: 67,
    12: 18,
    13: 19,
    14: 4,
    15: 1,
    16: 64,
    17: 20,
    18: 63,
    19: 7,
    20: 72,
    255: 255
}


class LabelTransform:
    def __init__(self, mapping):
        self.mapping = mapping

    def __call__(self, x):
        return Image.fromarray(self.mapping[x])


mapping = np.zeros((256,), dtype=np.uint8)
for k,v in coco_map.items():
    mapping[k] = v
remap = LabelTransform(mapping)

data = VOCSegmentation('data', train=True)
os.makedirs(f"data/voc/SegmentationClassAugAsCoco", exist_ok=True)

split_txt = []
for ip, lp in tqdm(data.images):
    lbl = Image.open(lp)
    idx = lp.split("/")[-1][:-4]
    new_path = f"data/voc/SegmentationClassAugAsCoco/{idx}.png"
    new_lbl = remap(lbl)
    new_lbl.save(new_path, "PNG")
