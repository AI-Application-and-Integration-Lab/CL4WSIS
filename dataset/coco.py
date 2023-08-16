import os.path as osp
import torch.utils.data as data
import numpy as np
from .dataset import IncrementalInstanceSegmentationDataset
from PIL import Image
from pycocotools.coco import COCO as COCOAPI

ignore_labels = [12, 26, 29, 30, 45, 66, 68, 69, 71, 83, 91]  # starting from 1=person


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

class COCO(data.Dataset):

    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 indices=None):
        
        root = osp.expanduser(root)
        base_dir = "coco"
        ds_root = osp.join(root, base_dir)
        splits_dir = osp.join(ds_root, 'split')
        self.ds_root = ds_root
        self.is_train = train

        if train:
            self.image_set = "train"
            split_f = osp.join(splits_dir, 'train.txt')
            folder = 'train2017'
            ann_f = 'instances_train2017.json'
        else:
            self.image_set = "val"
            split_f = osp.join(splits_dir, 'val.txt')
            folder = 'val2017'
            ann_f = 'instances_val2017.json'

        self.folder = folder
        ann_folder = "annotations"

        with open(osp.join(split_f), "r") as f:
            files = f.readlines()

        self.coco = COCOAPI(osp.join(ds_root, ann_folder, ann_f))
        self.ids = list(sorted(self.coco.imgs.keys()))
        
        self.images = [x[:-1] + ".jpg" for x in files]
        if indices is not None:
            self.images = [self.images[idx] for idx in indices]
        
        ids = []
        for img_id in self.ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
            anno = self.coco.loadAnns(ann_ids)
            if has_valid_annotation(anno):
                if self.coco.imgs[img_id]['file_name'] in self.images:
                    ids.append(img_id)
        
        # self.img_lvl_labels = np.load(osp.join(ds_root, f"1h_labels_{self.image_set}.npy"))

        self.transform = transform
        self.indices = ids
        
    def _load_image(self, id):
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(osp.join(self.ds_root, "images", self.folder, path)).convert("RGB"), path

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
        
        seg = Image.fromarray(np.max(np.stack([self.coco.annToMask(ann) * ann['category_id']
                                                for ann in anno]), axis=0))
        
        if not self.is_train:
            mask = np.stack([self.coco.annToMask(ann) for ann in anno])
            mask_label = np.array([ann['category_id'] for ann in anno])
            
            return img, seg, mask, mask_label, path
        
        mask = Image.fromarray(np.max(np.stack([self.coco.annToMask(ann) * (idx + 1)
                                                 for idx, ann in enumerate(anno)]), axis=0))
        img_lvl_lbls = np.zeros((91, ))
        img_lvl_lbls[np.unique([ann['category_id'] for ann in anno]) - 1] = 1
        
        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, seg, mask, img_lvl_lbls

    def __len__(self):
        return len(self.indices)


class COCOIncremental(IncrementalInstanceSegmentationDataset):
    def make_dataset(self, root, ann_file, old_classes, new_classes, is_train, val_on_trainset, pseudo=None, overlap=True, indices=None, masking=True):
        full_voc = COCO(root, is_train, transform=None, indices=indices)
        return full_voc

