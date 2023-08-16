from .voc import VOCSegmentation, VOCSegmentationIncremental, VOCasCOCOSegmentationIncremental, VOCInstanceSegmentationIncremental
from .coco import COCO, COCOIncremental
from .transform import *
import tasks
import os

from dataset.transforms import transforms as T

def get_dataset(opts):
    """ Dataset And Augmentation
    """

    train_transform = transform.Compose([
        transform.RandomResizedCrop(opts.crop_size, (0.5, 2)),
        transform.RandomHorizontalFlip(),
        transform.ToTensor(),
        transform.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])
    
    val_transform = transform.Compose([
        transform.Resize(size=opts.crop_size_val),
        transform.ToTensor(),
        transform.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
    test_transform = val_transform
    
    pam_transform = transform.Compose([
        transform.RandomResizedCrop(opts.crop_size, (0.5, 2)),
        transform.RandomHorizontalFlip(),
        transform.ToTensor(),
        transform.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])

    step_dict = tasks.get_task_dict(opts.dataset, opts.task, opts.step)
    labels, labels_old, path_base = tasks.get_task_labels(opts.dataset, opts.task, opts.step)

    pseudo = f"{opts.pseudo}" if opts.pseudo is not None else None
    path_base = os.path.join(opts.data_root, path_base)

    labels_cum = labels_old + labels
    masking_value = 0

    if opts.dataset == 'voc':
        t_dataset = dataset = VOCInstanceSegmentationIncremental
    elif opts.dataset == 'coco-voc':
        if opts.step == 0:
            t_dataset = dataset = COCOIncremental
        else:
            dataset = VOCasCOCOSegmentationIncremental
            t_dataset = COCOIncremental
    else:
        raise NotImplementedError

    if opts.overlap and opts.dataset == 'voc':
        path_base += "-ov"

    path_base_train = path_base

    if not os.path.exists(path_base):
        os.makedirs(path_base, exist_ok=True)

    train_dst = dataset(root=opts.data_root, step_dict=step_dict, train=True, transform=train_transform, pam_transform=pam_transform,
                        idxs_path=path_base_train + f"/train-{opts.step}.npy", masking_value=masking_value,
                        masking=not opts.no_mask, overlap=opts.overlap, step=opts.step, weakly=opts.weakly,
                        pseudo=pseudo, ann_file=opts.data_root + "/voc/pascal_sbd_train.json", )

    val_dst = dataset(root=opts.data_root, step_dict=step_dict, train=False, transform=val_transform,
                      idxs_path=path_base + f"/val-{opts.step}.npy", masking_value=masking_value,
                      masking=False, overlap=opts.overlap, step=opts.step, weakly=opts.weakly, ann_file=opts.data_root + "/voc/pascal_sbd_val.json")

    image_set = 'train' if opts.val_on_trainset else 'val'
    test_dst = t_dataset(root=opts.data_root, step_dict=step_dict, train=False, 
                         val_on_trainset=opts.val_on_trainset, transform=test_transform,
                         masking=False, masking_value=255, weakly=opts.weakly,
                         idxs_path=path_base + f"/test_on_{image_set}-{opts.step}.npy", step=opts.step, ann_file=opts.data_root + f"/voc/pascal_sbd_{image_set}.json")

    return train_dst, val_dst, test_dst, labels_cum, len(labels_cum)

