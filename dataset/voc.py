import os
import torch.utils.data as data
import torchvision

from .dataset import IncrementalSegmentationDataset, IncrementalInstanceSegmentationDataset
import numpy as np

from PIL import Image

classes = {
    0: 'background',
    1: 'aeroplane',
    2: 'bicycle',
    3: 'bird',
    4: 'boat',
    5: 'bottle',
    6: 'bus',
    7: 'car',
    8: 'cat',
    9: 'chair',
    10: 'cow',
    11: 'diningtable',
    12: 'dog',
    13: 'horse',
    14: 'motorbike',
    15: 'person',
    16: 'pottedplant',
    17: 'sheep',
    18: 'sofa',
    19: 'train',
    20: 'tvmonitor'
}
task_list = ['person', 'animals', 'vehicles', 'indoor']
tasks = {
    'person': [15],
    'animals': [3, 8, 10, 12, 13, 17],
    'vehicles': [1, 2, 4, 6, 7, 14, 19],
    'indoor': [5, 9, 11, 16, 18, 20]
}

coco_map = [1, 2, 3, 4, 5, 6, 7, 9, 16, 17, 18, 19, 20, 21, 44, 62, 63, 64, 67, 72]


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

        self.is_train = train
        self.pseudo = pseudo
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

        self.as_coco = as_coco
        
        if not os.path.exists(split_f):
            raise ValueError(
                'Wrong image_set entered! Please use image_set="train" '
                'or image_set="trainval" or image_set="val"')

        # remove leading \n
        with open(os.path.join(split_f), "r") as f:
            file_names = [x[:-1].split(' ') for x in f.readlines()]
        self.file_names = [x[0][1:].split('/')[1] for x in file_names]
        # REMOVE FIRST SLASH OTHERWISE THE JOIN WILL start from root
        ori_path = 'SegmentationClassAug'
        if as_coco:
            ori_path = 'SegmentationClassAugAsCoco'
        self.images = [(os.path.join(voc_root, x[0][1:]), os.path.join(voc_root, x[1][1:]), os.path.join(voc_root, x[1][1:].replace(ori_path, 'SegmentationObjectAug'))) for x in file_names]
        if saliency:
            self.saliency_images = [x[0].replace("JPEGImages", "SALImages")[:-3] + "png" for x in self.images]
        else:
            self.saliency_images = None

        if self.pseudo is not None and train:
            if not as_coco:
                self.images = [(x[0], x[1].replace("SegmentationClassAug", f"PseudoLabels/{self.pseudo}/rw/"), x[2]) for x in self.images]
            elif self.pseudo is not None:
                pseudo = self.pseudo
                self.images = [(x[0], x[1].replace("SegmentationClassAugAsCoco", f"pseudo_data/{pseudo}/seg_{pseudo}_AsCoco"), x[2]) for x in self.images]
            else:
                assert "no method"
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
        seg = Image.open(self.images[self.indices[index]][1])
        mask = Image.open(self.images[self.indices[index]][2])
        
        img_lvl_lbls = self.img_lvl_labels[self.indices[index]]
        file_name = self.file_names[self.indices[index]]
        name = file_name.split('.')[0]
        
        if not self.is_train:
            mask_arr = np.array(mask)
            seg_arr = np.array(seg)
            mask_ = []
            mask_label = []
            for idx in np.unique(mask_arr):
                if idx != 0 and idx != 255:
                    mask_.append((mask_arr == idx).astype(int))
                    cls, counts = np.unique(seg_arr[mask_arr == idx], return_counts=True)
                    assert len(cls) == 1, f"{cls}, {counts}, {self.images[self.indices[index]][1]}, {self.images[self.indices[index]][2]}"
                    mask_label.append(cls[0])
            mask = np.stack(mask_)
            mask_label = np.array(mask_label)
            
            return img, seg, mask, mask_label, name
        
        if self.pseudo is not None:
            pseudo = self.pseudo
            npy_path = f'data/voc/{pseudo}/ins_seg_{pseudo}/{name}.npy'
            npy_file = np.load(npy_path, allow_pickle=True).item()
            mask = []
            for i in range(npy_file['mask'].shape[0]):
                mask.append(npy_file['mask'][i].astype(np.uint8) * (i + 1))
            mask = Image.fromarray(np.max(np.stack(mask), axis=0))

        return img, seg, mask, img_lvl_lbls

    def __len__(self):
        return len(self.indices)

def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)

def has_valid_annotation(anno):
    # if it's empty, there is no annotation
    if len(anno) == 0:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False
    # keypoints task have a slight different critera for considering
    # if an annotation is valid
    if "keypoints" not in anno[0]:
        return True
    return False

def image_annotation(anno, classes):
    """
    only new categories' annotations
    """
    real_anno = []
    for i in anno:
        if i['category_id'] in classes:
            real_anno.append(i)
    return real_anno
    
def check_if_insert(anno, overlap, seen_classes, new_classes, is_train=True):
    if not is_train:
        return True
    
    if overlap:
        for i in anno:
            if i['category_id'] in new_classes:
                return True
        return False
    else: # disjoint
        is_new = False
        for i in anno:
            if i['category_id'] in new_classes: # new class in image
                is_new = True
            if i['category_id'] not in seen_classes: # future class in image
                return False
        return is_new

class VOCInstanceSegmentation(torchvision.datasets.coco.CocoDetection):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Segmentation Dataset.
    Args:
        root (string): Root directory of the VOC Dataset.
        image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val``
        is_aug (bool, optional): If you want to use the augmented train set or not (default is True)
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    """

    def __init__(self,
                 data_dir, 
                 ann_file,
                 old_classes,
                 new_classes,
                 is_train=True,
                 val_on_trainset=False,
                 pseudo=None,
                 overlap=True,
                 as_coco=False,
                 masking=True):

        super(VOCInstanceSegmentation, self).__init__(data_dir + "/voc/JPEGImages", ann_file)
        self.ids = sorted(self.ids)
        self.is_train = is_train
        self.old_classes = old_classes
        self.new_classes = new_classes
        self.pseudo = pseudo
        self.masking = masking
        self.val_on_trainset = val_on_trainset
        count = 0

        # filter images without detection annotations
        ids = []
        training = self.is_train
        if self.val_on_trainset:
            training = True
        for img_id in self.ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
            anno = self.coco.loadAnns(ann_ids)
            if has_valid_annotation(anno):
                if check_if_insert(anno, overlap, new_classes+old_classes, new_classes, training):  # filtering images for new categories
                    count = count + 1
                    ids.append(img_id)
        
        self.as_coco = as_coco
        self.indices = ids
        if self.is_train:
            print('number of images used for training: {0}'.format(count))
        else:
            print('number of images used for testing: {0}'.format(count))
        self.num_img = count

        self.img_lvl_labels = None

    def _load_image(self, id):
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB"), path

    def _load_target(self, id):
        return self.coco.loadAnns(self.coco.getAnnIds(id))
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        id = self.indices[index]
        img, path = self._load_image(id)
        anno = self._load_target(id)
        # filter annotation for old, new and exclude classes data
        if self.is_train and not self.val_on_trainset:
            if self.masking:
                anno = image_annotation(anno, self.new_classes)
        else:
            seg = Image.fromarray(np.max(np.stack([self.coco.annToMask(ann) * ann['category_id']
                                                 for ann in anno]), axis=0))
            mask = np.stack([self.coco.annToMask(ann) for ann in anno])
            mask_label = np.array([ann['category_id'] for ann in anno])
            
            return img, seg, mask, mask_label, path
        
        seg = Image.fromarray(np.max(np.stack([self.coco.annToMask(ann) * ann['category_id']
                                                 for ann in anno]), axis=0))
        mask = Image.fromarray(np.max(np.stack([self.coco.annToMask(ann) * (idx + 1)
                                                 for idx, ann in enumerate(anno)]), axis=0))
        
        img_lvl_lbls = np.zeros((len(self.old_classes) + len(self.new_classes), ))
        img_lvl_lbls[np.unique([ann['category_id'] for ann in anno]) - 1] = 1

        if self.pseudo is not None:
            name = path.split('.')[0]
            pseudo = self.pseudo
            npy_path = f'data/voc/{pseudo}/ins_seg_{pseudo}/{name}.npy'
            npy_file = np.load(npy_path, allow_pickle=True).item()
            mask = []
            seg = []
            for i in range(npy_file['mask'].shape[0]):
                mask.append(npy_file['mask'][i].astype(np.uint8) * (i + 1))
                seg.append(npy_file['mask'][i].astype(np.uint8) * (npy_file['class'][i] + 1))
                
            mask = Image.fromarray(np.max(np.stack(mask), axis=0))
            seg = Image.fromarray(np.max(np.stack(seg), axis=0))

        return img, seg, mask, img_lvl_lbls

    def __len__(self):
        return len(self.indices)

class VOCSegmentationIncremental(IncrementalSegmentationDataset):
    def make_dataset(self, root, train, indices, saliency=False, pseudo=None):
        full_voc = VOCSegmentation(root, train, transform=None, indices=indices, saliency=saliency, pseudo=pseudo)
        return full_voc

class VOCInstanceSegmentationIncremental(IncrementalInstanceSegmentationDataset):
    def make_dataset(self, root, ann_file, old_classes, new_classes, is_train, val_on_trainset, pseudo=None, overlap=True, indices=None, masking=True):
        full_voc = VOCInstanceSegmentation(root, ann_file, old_classes, new_classes, is_train, val_on_trainset, pseudo, overlap, masking=masking)
        return full_voc

class VOCasCOCOSegmentationIncremental(IncrementalInstanceSegmentationDataset):
    def make_dataset(self, root, ann_file, old_classes, new_classes, is_train, val_on_trainset,pseudo=None, overlap=True, indices=None, masking=True):
        full_voc = VOCSegmentation(root, is_train, transform=None, indices=indices, as_coco=True,
                                    saliency=None, pseudo=pseudo)
        return full_voc

class LabelTransform:
    def __init__(self, mapping):
        self.mapping = mapping

    def __call__(self, x):
        return Image.fromarray(self.mapping[x])
