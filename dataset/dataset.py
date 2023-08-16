import os
import torch.utils.data as data
from PIL import Image
from torch import from_numpy
import numpy as np
from .utils import gaussian, label_generation

class IncrementalSegmentationDataset(data.Dataset):
    def __init__(self,
                 root,
                 step_dict,
                 train=True,
                 transform=None,
                 idxs_path=None,
                 masking=True,
                 overlap=True,
                 masking_value=0,
                 step=0,
                 weakly=False,
                 pseudo=None):

        # take index of images with at least one class in labels and all classes in labels+labels_old+[255]
        if train:
            if idxs_path is not None and os.path.exists(idxs_path):
                idxs = np.load(idxs_path)
            else:
                raise FileNotFoundError(f"Please, add the traning spilt in {idxs_path}.")
        else:  # In both test and validation we want to use all data available (even if some images are all bkg)
            idxs = None

        self.dataset = self.make_dataset(root, train, indices=idxs, pseudo=pseudo)
        self.transform = transform
        self.weakly = weakly  # don't use weakly in val
        self.train = train

        self.step_dict = step_dict
        self.labels = []
        self.labels_old = []
        self.step = step

        self.order = [c for s in sorted(step_dict) for c in step_dict[s]]
        # assert not any(l in labels_old for l in labels), "Labels and labels_old must be disjoint sets"
        if step > 0:
            self.labels = [self.order[0]] + list(step_dict[step])
        else:
            self.labels = list(step_dict[step])
        self.labels_old = [lbl for s in range(step) for lbl in step_dict[s]]

        self.masking_value = masking_value
        self.masking = masking

        self.inverted_order = {lb: self.order.index(lb) for lb in self.order}
        if train:
            self.inverted_order[255] = masking_value
        else:
            self.set_up_void_test()

        if masking:
            tmp_labels = self.labels + [255]
            mapping_dict = {x: self.inverted_order[x] for x in tmp_labels}
        else:
            mapping_dict = self.inverted_order

        # if not (train and self.weakly):
        mapping = np.zeros((256,))
        for k in mapping_dict.keys():
            mapping[k] = mapping_dict[k]
        self.transform_lbl = LabelTransform(mapping)
        self.transform_1h = LabelSelection(self.order, self.labels, self.masking)

    def set_up_void_test(self):
        self.inverted_order[255] = 255

    def __getitem__(self, index):
        if index < 0:
            if -index > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            index = len(self) + index

        if index < len(self):
            data = self.dataset[index]
            img, lbl, lbl_1h = data[0], data[1], data[2]

            img, lbl = self.transform(img, lbl)
            lbl = self.transform_lbl(lbl)
            l1h = self.transform_1h(lbl_1h)
            return img, lbl, l1h

        else:
            raise ValueError("absolute value of index should not exceed dataset length")

    @staticmethod
    def __strip_zero(labels):
        while 0 in labels:
            labels.remove(0)

    def __len__(self):
        return len(self.dataset)

    def make_dataset(self, root, train, indices, saliency=False, pseudo=None):
        raise NotImplementedError


class LabelTransform:
    def __init__(self, mapping):
        self.mapping = mapping

    def __call__(self, x):
        return from_numpy(self.mapping[x])

class IncrementalInstanceSegmentationDataset(data.Dataset):
    def __init__(self,
                 root,
                 step_dict,
                 train=True,
                 val_on_trainset=False,
                 transform=None,
                 idxs_path=None,
                 masking=True,
                 overlap=True,
                 masking_value=0,
                 step=0,
                 sup='cls', 
                 sigma=8, 
                 point_thresh=0.5,
                 weakly=False,
                 pseudo=None,
                 ann_file=None):
        
        self.transform = transform
        self.weakly = weakly  # don't use weakly in val
        self.train = train

        self.step_dict = step_dict
        self.labels = []
        self.labels_old = []
        self.step = step
        self.sup = sup
        self.point_thresh = point_thresh
        self.sigma = sigma
        self.g = gaussian(sigma)
        self.pseudo = pseudo
        self.val_on_trainset = val_on_trainset

        self.order = [c for s in sorted(step_dict) for c in step_dict[s]]
        if step > 0:
            self.labels = [self.order[0]] + list(step_dict[step]) # bg + cur task classes
        else:
            self.labels = list(step_dict[step]) # first task already contain bg
        self.labels_old = [lbl for s in range(step) for lbl in step_dict[s]]
        self.total_classes = len(self.labels_old) + len(list(step_dict[step])) - 1 # without bg
        
        if train and ('coco' in idxs_path):
            if idxs_path is not None and os.path.exists(idxs_path):
                idxs = np.load(idxs_path)
            else:
                raise FileNotFoundError(f"Please, add the traning spilt in {idxs_path}.")
        else:  # In both test and validation we want to use all data available (even if some images are all bkg)
            idxs = None
        
        self.dataset = self.make_dataset(root, ann_file, old_classes=self.labels_old[1:], new_classes=self.labels[1:], is_train=train, val_on_trainset=val_on_trainset, pseudo=pseudo, overlap=overlap, indices=idxs, masking=masking)

        self.masking_value = masking_value
        self.masking = masking

        self.inverted_order = {lb: self.order.index(lb) for lb in self.order}
        if train:
            self.inverted_order[255] = masking_value
        else:
            self.set_up_void_test()

        if masking:
            tmp_labels = self.labels + [255]
            mapping_dict = {x: self.inverted_order[x] for x in tmp_labels}
        else:
            mapping_dict = self.inverted_order

        mapping = np.zeros((256,))
        for k in mapping_dict.keys():
            mapping[k] = mapping_dict[k]
        self.mapping = mapping
        self.transform_lbl = LabelTransform(mapping)
        self.transform_1h = LabelSelection(self.order, self.labels, self.masking)

    def set_up_void_test(self):
        self.inverted_order[255] = 255

    def __getitem__(self, index):
        if index < 0:
            if -index > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            index = len(self) + index

        if index < len(self):
            data = self.dataset[index]
            if not self.train or self.val_on_trainset:
                img, seg, mask, mask_label, path = data[0], data[1], data[2], data[3], data[4]
                img, seg = self.transform(img, seg)
                seg = self.transform_lbl(seg)
                mask_label = np.array([self.mapping[i] for i in mask_label])
                assert mask.shape[0] == len(mask_label)
                return img, seg, mask, mask_label, path
            
            img, seg_map, mask, lbl_1h = data[0], data[1], data[2], data[3]
            # seg_map: semantic segmentaion
            # mask: instance segmentation
            # lbl_1h: image labels

            label = self.concat_PIL(seg_map, mask)
            img, label = self.transform(img, label)
            
            seg_map, mask = label[:, :, 0], label[:, :, 1].numpy()
            seg_map = self.transform_lbl(seg_map)
            l1h = self.transform_1h(lbl_1h)
            
            mask *= (seg_map.numpy() > 0) # remove instances that are not in this task
            
            center_map, offset_map, weight = label_generation(seg_map.numpy(),
                                                              mask,
                                                              self.total_classes,
                                                              self.sigma,
                                                              self.g)
            
            return img, seg_map, center_map, offset_map, weight, l1h

        else:
            raise ValueError("absolute value of index should not exceed dataset length")

    def make_class_wise_point_list(self, points):
        
        MAX_NUM_POINTS = 128
        
        point_list = np.zeros((self.total_classes, MAX_NUM_POINTS, 2), dtype=np.int32)
        point_count = [0 for _ in range(self.total_classes)]
        
        for (x, y, cls, _ ) in points:
            if cls >= self.total_classes:
                continue
            point_list[cls][point_count[cls]] = [y, x]
            point_count[cls] += 1
            
        return point_list
    
    @staticmethod
    def __strip_zero(labels):
        while 0 in labels:
            labels.remove(0)

    def __len__(self):
        return len(self.dataset)
    
    def make_dataset(self, root, train, indices, saliency=False, pseudo=None):
        raise NotImplementedError

    def concat_PIL(self, x, y):
        x = np.array(x)[:, :, np.newaxis]
        y = np.array(y)[:, :, np.newaxis]
        result = np.concatenate([x, y], axis=-1)
        return Image.fromarray(np.uint8(result))

class LabelTransform:
    def __init__(self, mapping):
        self.mapping = mapping

    def __call__(self, x):
        return from_numpy(self.mapping[x])


class LabelSelection:
    def __init__(self, order, labels, masking):
        order = np.array(order)
        order = order[order != 0]
        order -= 1  # scale to match one-hot index.
        self.order = order
        if masking:
            self.masker = np.zeros((len(order)))
            self.masker[-len(labels)+1:] = 1
        else:
            self.masker = np.ones((len(order)))

    def __call__(self, x):
        # for coco-to-voc, len(x) is 91 and we use order to get the 80 classes we want in order 
        x = x[self.order] * self.masker
        return x
