from cv2 import norm
import torch
import numpy as np
import os
from torch import distributed
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
import tqdm
from utils.loss import KnowledgeDistillationLoss, BCEWithLogitsLossWithIgnoreIndex, \
    UnbiasedKnowledgeDistillationLoss, UnbiasedCrossEntropy, IcarlLoss, DeepLabCE, Weighted_L1_Loss, Weighted_MSELoss
from torch.cuda import amp
from segmentation_module import make_model, TestAugmentation
import tasks
from torch.nn.parallel import DistributedDataParallel, DataParallel
import os.path as osp
from wss.modules import PAMR, ASPP, PeakGenerator, PseudoLabeler
from utils.utils import denorm, label_to_one_hot
from wss.single_stage import pseudo_gtmask, balanced_mask_loss_ce, balanced_mask_loss_unce
from wss.utils import peak_extract, smoothing
from utils.wss_loss import bce_loss, ngwp_focal, binarize, RandropLoss
from segmentation_module import get_norm
from utils.scheduler import get_scheduler
from dataset.utils import get_ins_map, ppmg
from modules.utils import refine_label_generation, pseudo_label_generation
from torch.distributions import Categorical

from collections import defaultdict
from metrics import eval_instance_segmentation_voc
from chainercv.utils.mask.mask_iou import mask_iou
from dataset.utils import gaussian
from utils.utils import label_to_color_image
import cv2

from torchvision.transforms.functional import rotate
from torch import distributed

class Trainer:
    def __init__(self, logger, device, opts):
        self.logger = logger
        self.device = device
        self.opts = opts
        self.scaler = amp.GradScaler()

        self.classes = classes = tasks.get_per_task_classes(opts.dataset, opts.task, opts.step)

        if classes is not None:
            new_classes = classes[-1]
            self.tot_classes = reduce(lambda a, b: a + b, classes)
            self.old_classes = self.tot_classes - new_classes
        else:
            self.old_classes = 0

        self.model = make_model(opts, classes=classes)

        if opts.step == 0:  # if step 0, we don't need to instance the model_old
            self.model_old = None
        else:  # instance model_old
            self.model_old = make_model(opts, classes=tasks.get_per_task_classes(opts.dataset, opts.task, opts.step - 1))
            self.model_old.to(self.device)
            # freeze old model and set eval mode
            for par in self.model_old.parameters():
                par.requires_grad = False
            self.model_old.eval()

        self.weakly = opts.weakly and opts.step > 0
        self.pos_w = opts.pos_w
        self.use_aff = opts.affinity
        self.weak_single_stage_dist = opts.ss_dist
        self.pseudo_epoch = opts.pseudo_ep
        cls_classes = self.tot_classes
        self.pseudolabeler = None
        self.peakgenerator = None
        
        self.peak_conf_thresh = opts.pseudo_thresh
        self.sigma = opts.sigma
        self.g = gaussian(self.sigma)

        if self.weakly and opts.pseudo is None:
            if opts.affinity_method == 'pamr':
                self.affinity = PAMR(num_iter=10, dilations=[1, 2, 4, 8, 12]).to(device)
            for p in self.affinity.parameters():
                p.requires_grad = False
            norm = get_norm(opts)
            channels = 4096 if "wide" in opts.backbone else 2048
            self.pseudolabeler = PseudoLabeler(channels, cls_classes, norm)

            if self.opts.peak_from == 'peakgenerator':
                self.peakgenerator = PeakGenerator(in_channel=cls_classes, num_classes=cls_classes - 1, old_classes=self.old_classes - 1)
            
            self.icarl = opts.icarl
        
        self.optimizer, self.scheduler = self.get_optimizer(opts)

        self.distribute(opts)

        # Select the Loss Type
        reduction = 'none'

        self.center_loss_weight = 200
        self.offset_loss_weight = 0.01
        self.bce = opts.bce or opts.icarl
        if self.bce:
            print('using bce')
            seg_loss = BCEWithLogitsLossWithIgnoreIndex(reduction=reduction)
        elif self.opts.dce:
            print('using DeepLabCE')
            seg_loss = DeepLabCE(ignore_label=255, top_k_percent_pixels=0.2)
        else:
            seg_loss = nn.BCEWithLogitsLoss()
        n_gpus = torch.cuda.device_count()
        self.criterion = {
            'seg': seg_loss,
            'center': Weighted_MSELoss(),
            'offset': Weighted_L1_Loss()
        }
        self.randrop = RandropLoss(self.opts.epochs - self.pseudo_epoch, self.tot_classes, self.old_classes, self.device)

        # ILTSS
        self.lde = opts.loss_de
        self.lde_flag = self.lde > 0. and self.model_old is not None
        self.lde_loss = nn.MSELoss()

        self.lkd = opts.loss_kd
        self.lkd_flag = self.lkd > 0. and self.model_old is not None
        if opts.unkd:
            self.lkd_loss = UnbiasedKnowledgeDistillationLoss(alpha=opts.alpha)
        else:
            self.lkd_loss = KnowledgeDistillationLoss(alpha=opts.alpha)

        # ICARL
        self.icarl_combined = False
        self.icarl_only_dist = False
        if opts.icarl:
            self.icarl_combined = not opts.icarl_disjoint and self.model_old is not None
            self.icarl_only_dist = opts.icarl_disjoint and self.model_old is not None
            if self.icarl_combined:
                self.licarl = nn.BCEWithLogitsLoss(reduction='mean')
                self.icarl = opts.icarl_importance
            elif self.icarl_only_dist:
                self.licarl = IcarlLoss(reduction='mean', bkg=opts.icarl_bkg)
        self.icarl_dist_flag = self.icarl_only_dist or self.icarl_combined
        
    def get_optimizer(self, opts):
        params = []
        if not opts.freeze:
            params.append({"params": filter(lambda p: p.requires_grad, self.model.body.parameters()),
                           'weight_decay': opts.weight_decay})

        if opts.phase is None or opts.phase == 2:
            if hasattr(self.model, 'decoder'):
                if hasattr(self.model.decoder, 'semantic_decoder'):
                    params.append({"params": filter(lambda p: p.requires_grad, self.model.decoder.semantic_decoder.parameters()),
                                    'weight_decay': opts.weight_decay, 'lr': opts.lr*opts.lr_head})
                if hasattr(self.model.decoder, 'instance_decoder'):
                    params.append({"params": filter(lambda p: p.requires_grad, self.model.decoder.instance_decoder.parameters()),
                                    'weight_decay': opts.weight_decay, 'lr': opts.lr*opts.lr_head})
            
            if self.model.instance_head is not None:
                params.append({"params": filter(lambda p: p.requires_grad, self.model.instance_head.parameters()),
                            'weight_decay': opts.weight_decay, 'lr': opts.lr*opts.lr_head})
        if not opts.freeze_seg:
            if self.model.head is not None:
                params.append({"params": filter(lambda p: p.requires_grad, self.model.head.parameters()),
                                'weight_decay': opts.weight_decay, 'lr': opts.lr*opts.lr_head})
                params.append({"params": filter(lambda p: p.requires_grad, self.model.cls.parameters()),
                                'weight_decay': opts.weight_decay, 'lr': opts.lr*opts.lr_head})
            else:
                params.append({"params": filter(lambda p: p.requires_grad, self.model.semantic_head.parameters()),
                                'weight_decay': opts.weight_decay, 'lr': opts.lr*opts.lr_head})

            if self.weakly and self.opts.pseudo is None:
                params.append({"params": filter(lambda p: p.requires_grad, self.pseudolabeler.parameters()),
                            'weight_decay': opts.weight_decay, 'lr': opts.lr_pseudo})
                if self.peakgenerator is not None:
                    params.append({"params": filter(lambda p: p.requires_grad, self.peakgenerator.parameters()),
                                'weight_decay': opts.weight_decay, 'lr': opts.lr_pseudo})

        if opts.optim == 'sgd':
            optimizer = torch.optim.SGD(params, lr=opts.lr, momentum=0.9, nesterov=True)
        elif opts.optim == 'adam':
            optimizer = torch.optim.Adam(params, lr=opts.lr, weight_decay=opts.weight_decay)
        scheduler = get_scheduler(opts, optimizer)

        return optimizer, scheduler
    
    def distribute(self, opts):
        self.model = DistributedDataParallel(self.model.to(self.device), device_ids=[opts.device_id],
                                             output_device=opts.device_id, find_unused_parameters=False)

        if self.weakly and self.opts.pseudo is None:
            self.pseudolabeler = DistributedDataParallel(self.pseudolabeler.to(self.device), device_ids=[opts.device_id],
                                                         output_device=opts.device_id, find_unused_parameters=False)
            if self.peakgenerator is not None:
                self.peakgenerator = DistributedDataParallel(self.peakgenerator.to(self.device), device_ids=[opts.device_id],
                                                         output_device=opts.device_id, find_unused_parameters=False)
    
    def train(self, cur_epoch, train_loader, print_int=10):
        """Train and return epoch loss"""
        optim = self.optimizer
        scheduler = self.scheduler
        device = self.device
        model = self.model
        center_loss_weight = self.center_loss_weight
        offset_loss_weight = self.offset_loss_weight
        criterion = self.criterion
        logger = self.logger

        logger.info("Epoch %d, lr = %f" % (cur_epoch, optim.param_groups[0]['lr']))

        epoch_loss = 0.0
        reg_loss = 0.0
        l_cam_out = 0.0
        l_cam_int = 0.0
        l_seg = 0.0
        l_center = 0.0
        l_offset = 0.0
        l_cls = 0.0
        interval_loss = 0.0

        lkd = torch.tensor(0.)
        lde = torch.tensor(0.)
        l_icarl = torch.tensor(0.)
        l_reg = torch.tensor(0.)
                
        train_loader.sampler.set_epoch(cur_epoch)
        
        if distributed.get_rank() == 0:
            tq = tqdm.tqdm(total=len(train_loader))
            tq.set_description("Epoch %d, lr = %f" % (cur_epoch, optim.param_groups[0]['lr']))
        else:
            tq = None
        
        model.train()
        if self.opts.phase == 2:
            if self.opts.freeze:
                model.module.body.eval()
            if self.opts.freeze_seg:
                model.module.head.eval()
                model.module.cls.eval()
        for cur_step, (images, seg_maps, center_maps, offset_maps, weights, l1h) in enumerate(train_loader):
            
            images = images.to(device, dtype=torch.float)
            seg_maps = seg_maps.to(device, dtype=torch.long)
            center_maps = center_maps.to(device, dtype=torch.float)
            offset_maps = offset_maps.to(device, dtype=torch.float)
            weights = weights.to(device, dtype=torch.float)
            l1h = l1h.to(device, dtype=torch.float)
            
            bs = images.shape[0]

            with amp.autocast():
                if (self.lde_flag or self.lkd_flag or self.icarl_dist_flag or self.weakly) and self.model_old is not None:
                    with torch.no_grad():
                        outputs_old, features_old = self.model_old(images, interpolate=False)
                    
                optim.zero_grad()
                
                if self.opts.phase == 2:
                    with torch.no_grad():
                        outputs, features = model.module.forward_seg(torch.cat([images, images.flip(-1)], dim=0), interpolate=False)
                    
                    outputs_seg_max = (outputs['seg'][:bs] + outputs['seg'][bs:].flip(-1)) / 2
                    outputs['seg'] = outputs['seg'][:bs]
                    features_ = {}
                    for key in features['features']:
                        features_[key] = features['features'][key][:bs].detach().clone()
                    
                    _, instance_l = self.model.module.decoder(features_)
                    instance = self.model.module.instance_head(instance_l)
                    for key in instance.keys():
                        outputs[key] = instance[key]
                else:
                    outputs, features = model(images, interpolate=False)
                    
                if self.opts.model == 'DeeplabV3':
                    outputs = {'seg': outputs}
                
                # xxx BCE / Cross Entropy Loss
                if not self.weakly:
                    for key in outputs:
                        outputs[key] = F.interpolate(outputs[key], size=images.shape[-2:], mode="bilinear", align_corners=False)
                    
                    if self.opts.phase == 1:
                        seg_loss = criterion['seg'](outputs['seg'], seg_maps) # B x H x W
                        loss = seg_loss
                        if self.bce:
                            loss = loss.mean()
                    elif self.opts.phase == 2:
                        seg_loss = 0 * outputs['seg'][0, 0].mean()
                        center_loss = criterion['center'](outputs['center'], center_maps, weights) * center_loss_weight # 200
                        offset_loss = criterion['offset'](outputs['offset'], offset_maps, weights) * offset_loss_weight # 0.01
                        loss = seg_loss + center_loss + offset_loss
                    else:
                        seg_loss = criterion['seg'](outputs['seg'], seg_maps) # B x H x W
                        center_loss = criterion['center'](outputs['center'], center_maps, weights) * center_loss_weight # 200
                        offset_loss = criterion['offset'](outputs['offset'], offset_maps, weights) * offset_loss_weight # 0.01
                        if self.bce:
                            seg_loss = seg_loss.mean()
                        loss = seg_loss + center_loss + offset_loss
                
                else:
                    if self.opts.phase == 1:
                        pseudolabeler_eval = self.pseudolabeler
                        pseudolabeler_eval.eval()
                        
                        feat_body = features['body'][:bs]
                        feat_body_eval = features['body']
                            
                        int_masks = pseudolabeler_eval(feat_body_eval)
                        int_masks = int_masks.detach()
                            
                        self.pseudolabeler.train()
                        if self.opts.flac:
                            angles = [90, 180, 270]
                            angle = angles[np.random.randint(3)]
                            feat_body = torch.cat([feat_body, feat_body.flip(-1), rotate(feat_body, angle=angle)], dim=0)
                            
                        int_masks_raw = self.pseudolabeler(feat_body)
                        
                        int_masks_raw2 = int_masks_raw
                        l1h2 = l1h
                        if self.opts.flac:
                            A = int_masks_raw
                            A_rot = torch.sigmoid(torch.mean(A[bs * 2: bs * 3, self.old_classes:], dim=1))
                            A_rot_back = rotate(A_rot.detach().clone(), angle=-angle)
                            A_flip = torch.sigmoid(torch.mean(A[bs: bs * 2, self.old_classes:].flip(-1), dim=1))
                            A_ori = torch.sigmoid(torch.mean(A[:bs, self.old_classes:], dim=1))
                                
                            A_target = torch.maximum(A_ori, A_flip).detach()
                            A_target = torch.maximum(A_target, A_rot_back).detach()
                            A_rot_target = rotate(A_target.clone(), angle=angle).detach()     
                            flac_loss = (self.lde_loss(A_ori, A_target) + self.lde_loss(A_flip, A_target) + self.lde_loss(A_rot, A_rot_target)) / 3
                            
                            int_masks_raw2 = int_masks_raw2[:bs]
                            
                            int_masks_raw = int_masks_raw[:bs]
                            int_masks = int_masks[:bs]
                        
                        if self.peakgenerator is not None:
                            self.peakgenerator.train()
                            peak_logits, _ = self.peakgenerator(int_masks_raw2)
                        
                        if self.opts.no_mask:
                            l_cam_new = bce_loss(int_masks_raw2, l1h2, mode=self.opts.cam, reduction='mean') 
                        else:
                            l_cam_new = bce_loss(int_masks_raw2, l1h2[:, self.old_classes - 1:],  mode=self.opts.cam, reduction='mean')
                            
                        if self.peakgenerator is not None:
                            l_peak_cam_new = F.binary_cross_entropy_with_logits(peak_logits[:, self.old_classes - 1:], l1h2[:, self.old_classes - 1:])
                        outputs_old_seg = F.interpolate(outputs_old['seg'], size=int_masks.shape[-2:], mode="bilinear", align_corners=False)
                        outputs_seg = F.interpolate(outputs['seg'], size=int_masks.shape[-2:], mode="bilinear", align_corners=False)
                        
                        target = torch.sigmoid(outputs_old_seg.detach())
                        if self.opts.no_mask:
                            target[:, 1:] *= l1h[:, :self.old_classes - 1, None, None]
                        l_loc = F.binary_cross_entropy_with_logits(int_masks_raw[:, :self.old_classes],
                                                            target,
                                                            reduction='mean')
                        
                        l_cam_int = l_cam_new + l_loc
                        if self.opts.flac:
                            l_cam_int += flac_loss
                        if self.peakgenerator is not None:
                            l_cam_int += l_peak_cam_new
                        
                        if self.lde_flag:
                            lde = self.lde * self.lde_loss(features['body'], features_old['body'])
                    
                        l_cam_out = 0 * outputs['seg'][0, 0].mean()  # avoid errors due to DDP
                        if self.opts.branch == 'all' or self.opts.branch == 'ins':
                            l_cam_out = l_cam_out + 0 * outputs['center'][0, 0].mean()  # avoid errors due to DDP
                            l_cam_out = l_cam_out + 0 * outputs['offset'][0, 0].mean()  # avoid errors due to DDP
                        
                    if self.opts.phase == 1 and cur_epoch >= self.pseudo_epoch:
                        
                        int_masks_orig = int_masks.softmax(dim=1)
                        int_masks_soft = int_masks.softmax(dim=1)

                        if self.use_aff:
                            image_raw = denorm(images)
                            im = F.interpolate(image_raw, int_masks.shape[-2:], mode="bilinear",
                                            align_corners=True)
                            int_masks_soft = self.affinity(im, int_masks_soft.detach())

                        int_masks_orig[:, 1:] *= l1h[:, :, None, None]
                        int_masks_soft[:, 1:] *= l1h[:, :, None, None]

                        pseudo_gt_seg = pseudo_gtmask(int_masks_soft, ambiguous=True, cutoff_top=0.6,
                                                    cutoff_bkg=0.7, cutoff_low=0.2).detach()

                        pseudo_gt_seg_lx = binarize(int_masks_orig)
                        pseudo_gt_seg_lx = (self.opts.alpha * pseudo_gt_seg_lx) + \
                                        ((1-self.opts.alpha) * int_masks_orig)

                        # ignore_mask = (pseudo_gt_seg.sum(1) > 0)
                        px_cls_per_image = pseudo_gt_seg_lx.view(bs, self.tot_classes, -1).sum(dim=-1)
                        batch_weight = torch.eq((px_cls_per_image[:, self.old_classes:] > 0),
                                                l1h[:, self.old_classes - 1:].bool())
                        batch_weight = (
                                    batch_weight.sum(dim=1) == (self.tot_classes - self.old_classes)).float()

                        target_old = torch.sigmoid(outputs_old_seg.detach())
                        
                        pseudo_seg_map = torch.cat((target_old, pseudo_gt_seg_lx[:, self.old_classes:]), dim=1)
                        if self.opts.icarl_bkg == -1:
                            pseudo_seg_map[:, 0] = torch.min(pseudo_seg_map[:, 0], pseudo_gt_seg_lx[:, 0])
                        else:
                            pseudo_seg_map[:, 0] = (1-self.opts.icarl_bkg) * pseudo_seg_map[:, 0] + \
                                        self.opts.icarl_bkg * pseudo_gt_seg_lx[:, 0]

                        l_seg = F.binary_cross_entropy_with_logits(outputs_seg, pseudo_seg_map, reduction='none').sum(dim=1)
                        l_seg = l_seg.view(bs, -1).mean(dim=-1)
                        l_seg = self.opts.l_seg * (batch_weight * l_seg).sum() / (batch_weight.sum() + 1e-5)
                        
                        l_cls = balanced_mask_loss_ce(int_masks_raw, pseudo_gt_seg, l1h)
                        
                        if self.opts.randrop:
                            int_masks_ref = torch.sigmoid(int_masks).clone()
                            int_masks_ref[:, 1:] *= l1h[:, :, None, None]
                            l_cam_int += self.randrop(int_masks_raw, int_masks_ref, cur_epoch - self.pseudo_epoch, label=l1h if self.opts.no_mask else None)

                    if self.opts.phase == 2:
                        """ center / offset generation """
                        
                        if self.peakgenerator is not None:
                            feat_body_eval = features['body']
                            
                            """from pseudolabeler"""
                            self.pseudolabeler.eval()
                            int_masks = self.pseudolabeler(feat_body_eval[:bs])
                            self.peakgenerator.eval()
                            _, cam = self.peakgenerator(int_masks, l1h)
                            cam = smoothing(cam)
                            
                            cam = F.interpolate(cam, size=images.shape[-2:], mode="bilinear", align_corners=False)

                        outputs['center'] = F.interpolate(outputs['center'], size=images.shape[-2:], mode="bilinear", align_corners=False)
                        outputs['offset'] = F.interpolate(outputs['offset'], size=images.shape[-2:], mode="bilinear", align_corners=False)

                        peak_conf, peak_y, peak_x = peak_extract(cam.detach(), kernel=15)
                        
                        outputs_seg_max = F.interpolate(outputs_seg_max, size=images.shape[-2:], mode="bilinear", align_corners=False).detach()
                        soft_seg_gt = outputs_seg_max.softmax(dim=1).detach()
                        if self.opts.no_mask:
                            soft_seg_gt[:, 1:] *= l1h[:, :, None, None]
                        else:
                            soft_seg_gt[:, self.old_classes:] *= l1h[:, self.old_classes - 1:, None, None]
                        seg_gt = torch.argmax(soft_seg_gt, 1)
                        
                        old_pseudo_weight = ((seg_gt < self.old_classes) & (seg_gt != 0))[:, None, :, :].float() # Bx1xHxW: foreground of old classes
                        seg_gt[seg_gt < self.old_classes] = 0
                        cls_label = l1h.cpu().detach().numpy() # BxC
                        cls_label[:, :self.old_classes - 1] = 0
                        
                        for b in range(bs):
                            points = []
                            valid_label = np.nonzero(cls_label[b])[0]
                            for l in valid_label:
                                for conf, x, y in zip(peak_conf[b, l], peak_x[b, l], peak_y[b, l]):
                                    if conf < self.opts.pseudo_thresh:
                                        break
                                    points.append([x, y, l, conf])
                            center_map, offset_map, weight, _ = pseudo_label_generation(seg_gt[b].cpu().numpy(),
                                            points, # points in image[b]
                                            cls_label[b], 
                                            self.tot_classes - 1, # remove bg class
                                            self.sigma, 
                                            self.g)
                            
                            center_map = torch.from_numpy(center_map)
                            offset_map = torch.from_numpy(offset_map)
                            weight = torch.from_numpy(weight)
                            
                            try:
                                pseudo_center_map = torch.cat([pseudo_center_map, center_map[None, :, :, :]], dim=0)
                                pseudo_offset_map = torch.cat([pseudo_offset_map, offset_map[None, :, :, :]], dim=0)
                                pseudo_weight = torch.cat([pseudo_weight, weight[None, :, :, :]], dim=0)
                            except:
                                pseudo_center_map = center_map[None, :, :, :]
                                pseudo_offset_map = offset_map[None, :, :, :]
                                pseudo_weight = weight[None, :, :, :]
                        
                        pseudo_center_map = pseudo_center_map.to(device, dtype=torch.float)
                        pseudo_offset_map = pseudo_offset_map.to(device, dtype=torch.float)
                        pseudo_weight = pseudo_weight.to(device, dtype=torch.float)
                        
                        run_refine = self.opts.run_refine
                        
                        if run_refine:
                            outputs_seg = outputs_seg_max
                            
                            left_seg_gt = seg_gt.clone()
                            
                            l1h_new = l1h.clone().detach()
                            l1h_new[:, :self.old_classes - 1] = 0
                            refined_label = refine_label_generation(
                                outputs_seg.clone().detach(), 
                                outputs['center'].clone().detach(), 
                                outputs['offset'].clone().detach(), 
                                l1h_new.clone().detach(), 
                                left_seg_gt.clone().detach(),
                                10000 if self.opts.task == 'voc' else None,
                                self.opts,
                            )
                        
                        # output_resize
                        outputs_old['center'] = F.interpolate(outputs_old['center'], size=images.shape[-2:], mode="bilinear", align_corners=False)
                        outputs_old['offset'] = F.interpolate(outputs_old['offset'], size=images.shape[-2:], mode="bilinear", align_corners=False)

                        pseudo_weight_sum = torch.maximum(old_pseudo_weight, pseudo_weight)
                        
                        if run_refine:
                            pseudo_center_map[:, self.old_classes - 1:] = pseudo_weight * pseudo_center_map[:, self.old_classes - 1:] + (1 - pseudo_weight) * refined_label['center'][:, self.old_classes - 1:]
                            pseudo_offset_map = pseudo_weight_sum * pseudo_offset_map + (1 - pseudo_weight_sum) * refined_label['offset']
                            pseudo_weight = torch.maximum(pseudo_weight, refined_label['weight'])
                        
                        # old classes
                        center_loss_1 = 0.5 * criterion['center'](outputs['center'][:, :self.old_classes - 1], outputs_old['center'], old_pseudo_weight) * center_loss_weight
                        offset_loss_1 = 0.5 * criterion['offset'](outputs['offset'], outputs_old['offset'], old_pseudo_weight) * offset_loss_weight
                        
                        # new classes
                        center_loss_2 = 0.5 * criterion['center'](outputs['center'][:, self.old_classes - 1:], pseudo_center_map[:, self.old_classes - 1:], pseudo_weight) * center_loss_weight
                        offset_loss_2 = 0.5 * criterion['offset'](outputs['offset'], pseudo_offset_map, pseudo_weight) * offset_loss_weight

                        l_center = center_loss_1 + center_loss_2
                        l_offset = offset_loss_1 + offset_loss_2
                    
                    if self.opts.freeze_seg:
                        l_seg = l_seg * 0
                        l_cls = l_cls * 0
                        l_cam_int = l_cam_int * 0
                    loss = l_seg + l_center + l_offset + l_cam_out
                    l_reg = l_cls + l_cam_int

                # xxx first backprop of previous loss (compute the gradients for regularization methods)
                if self.opts.freeze:
                    lde = lde * 0
                loss_tot = loss + lkd + lde + l_icarl + l_reg
                assert not torch.isnan(loss_tot) and not torch.isinf(loss_tot), f"l_reg: {loss_tot.item()}"
            
            self.scaler.scale(loss_tot).backward()
            self.scaler.step(optim)
            if scheduler is not None:
                scheduler.step()
            self.scaler.update()

            epoch_loss += loss.item()
            reg_loss += l_reg.item() if l_reg != 0. else 0.
            reg_loss += lkd.item() + lde.item() + l_icarl.item()
            interval_loss += loss.item() + lkd.item() + lde.item() + l_icarl.item()
            interval_loss += l_reg.item() if l_reg != 0. else 0.
            if tq is not None:
                tq.update(1)
                tq.set_postfix({'loss': '%.6f' % loss})
            
            if (cur_step + 1) % print_int == 0:
                interval_loss = interval_loss / print_int
                logger.debug(f"Epoch {cur_epoch}, Batch {cur_step + 1}/{len(train_loader)},"
                             f" Loss={interval_loss}")
                logger.debug(f"Loss made of: CE {loss}, LKD {lkd}, LDE {lde}, LReg {l_reg}")
                # visualization
                if logger is not None:
                    x = cur_epoch * len(train_loader) + cur_step + 1
                    logger.add_scalar('Loss/tot', interval_loss, x, intermediate=True)
                    logger.add_scalar('Loss/CAM_int', l_cam_int, x, intermediate=True)
                    logger.add_scalar('Loss/CAM_out', l_cam_out, x, intermediate=True)
                    logger.add_scalar('Loss/SEG_int', l_cls, x, intermediate=True)
                    logger.add_scalar('Loss/SEG_out', l_seg, x, intermediate=True)
                    logger.commit(intermediate=True)
                interval_loss = 0.0

        if tq is not None:
            tq.close()

        # collect statistics from multiple processes
        epoch_loss = torch.tensor(epoch_loss).to(self.device)
        reg_loss = torch.tensor(reg_loss).to(self.device)

        torch.distributed.reduce(epoch_loss, dst=0)
        torch.distributed.reduce(reg_loss, dst=0)

        if distributed.get_rank() == 0:
            epoch_loss = epoch_loss / distributed.get_world_size() / len(train_loader)
            reg_loss = reg_loss / distributed.get_world_size() / len(train_loader)
        
        loss_info = f"Epoch {cur_epoch}, Class Loss={epoch_loss}, Reg Loss={reg_loss}"
        logger.info(loss_info)
        # print(f"Epoch {cur_epoch}, Class Loss={epoch_loss}, Reg Loss={reg_loss}")

        return (epoch_loss, reg_loss)
    
    def validate(self, valid_loader, opts):
        
        iou_thresholds = np.arange(0.5, 0.95, 0.05).tolist()
        n_pos = [defaultdict(int) for _ in iou_thresholds]
        score = [defaultdict(list) for _ in iou_thresholds]
        match = [defaultdict(list) for _ in iou_thresholds]
        
        model = self.model
        device = self.device
        model.eval()
    
        val_dir = "tmp/val_temp_dir"
        if opts.local_rank == 0:
            os.makedirs(val_dir, exist_ok=True)
        
        i = -1
        count = 0
        torch.distributed.barrier()
        with torch.no_grad():
            for img, seg, mask, mask_label, fname in tqdm.tqdm(valid_loader):
                i += 1
                gt_mask = mask.numpy()[0]
                gt_label = (mask_label.numpy()[0] - 1) # -1 for removing background class, class in 
                
                target_size = mask.shape[-2:] # img is resized, but mask is in the original size
                
                if opts.val_flip:
                    img = torch.cat( [img, img.flip(-1)] , dim=0)
                    
                out, _ = model(img.to(device))
                    
                for key in out:
                    out[key] = F.interpolate(out[key], size=target_size, mode="bilinear", align_corners=False)
                
                pred_seg, pred_label, pred_mask, pred_score = get_ins_map(out, 
                                                                        False, 
                                                                        target_size, 
                                                                        device,
                                                                        opts)
                
                for idx, iou_thresh in enumerate(iou_thresholds):
                    n_pos[idx], score[idx], match[idx] = self.eval_detection_voc(n_pos[idx], score[idx], match[idx], gt_label, gt_mask, pred_label, pred_mask, pred_score, iou_thresh)
            
        torch.distributed.barrier()
        
        ap_masks = np.zeros((len(iou_thresholds), self.tot_classes - 1))
        
        ap_result = {}
        if opts.local_rank == 0:
            for idx in range(len(iou_thresholds)):
                tmp = eval_instance_segmentation_voc(n_pos[idx], score[idx], match[idx])['ap']
                ap_masks[idx] = tmp

            print("ap@.5", ap_masks[0], np.nanmean(ap_masks[0]))
            ap_05_95_mask = np.nanmean(ap_masks, axis=0)
            ap_result = {'ap': ap_05_95_mask, 'map': np.nanmean(ap_05_95_mask)} # mAP@0.5:0.95

            os.system(f"rm -rf {val_dir}")
            
        torch.distributed.barrier()
        
        model.train()
        
        return ap_result

    def eval_detection_voc(self, n_pos, score, match, gt_label, gt_mask, pred_label, pred_mask, pred_score, iou_thresh=0.5):
        
        for l in np.unique(np.concatenate((pred_label, gt_label)).astype(int)):
            pred_keep_l = pred_label == l
            pred_mask_l = pred_mask[pred_keep_l]
            pred_score_l = pred_score[pred_keep_l]
            # sort by score
            order = pred_score_l.argsort()[::-1]
            pred_mask_l = pred_mask_l[order]
            pred_score_l = pred_score_l[order]

            gt_keep_l = gt_label == l
            gt_mask_l = gt_mask[gt_keep_l]

            n_pos[l] += gt_keep_l.sum()
            score[l].extend(pred_score_l)

            if len(pred_mask_l) == 0:
                continue
            if len(gt_mask_l) == 0:
                match[l].extend((0,) * pred_mask_l.shape[0])
                continue

            iou = mask_iou(pred_mask_l, gt_mask_l)
            gt_index = iou.argmax(axis=1)
            # set -1 if there is no matching ground truth
            gt_index[iou.max(axis=1) < iou_thresh] = -1
            del iou

            selec = np.zeros(gt_mask_l.shape[0], dtype=bool)
            for gt_idx in gt_index:
                if gt_idx >= 0:
                    if not selec[gt_idx]:
                        match[l].append(1)
                    else:
                        match[l].append(0)
                    selec[gt_idx] = True
                else:
                    match[l].append(0)
                    
        return n_pos, score, match
    
    def validate_semseg(self, loader, metrics, opts, eval_pseudolabeler=False):
        """Do validation and return specified samples"""
        metrics.reset()
        model = self.model
        device = self.device
        
        if eval_pseudolabeler:
            pseudolabeler = self.pseudolabeler
            pseudolabeler.eval()
        model.eval()

        def classify(images):
            if eval_pseudolabeler:
                masks, _ = pseudolabeler(model(images, as_feature_extractor=True)['body'])
            else:
                masks, _ = model(images)
                if type(masks) is not torch.tensor:
                    masks = masks['seg']
            if opts.val_flip:
                # masks = torch.maximum(masks[:1], masks[1:].flip(-1))
                masks = (masks[:1] + masks[1:].flip(-1)) / 2.
            masks = F.interpolate(masks, size=images.shape[-2:], mode="bilinear", align_corners=False)
            masks = masks.softmax(dim=1)
            return masks

        i = -1
        with torch.no_grad():
            for img, seg, mask, mask_label, fname in tqdm.tqdm(loader):
                i = i+1
                images = img.to(device, dtype=torch.float32)
                labels = seg.to(device, dtype=torch.long)
                l1h = mask_label.to(device, dtype=torch.bool)

                with amp.autocast():
                    if opts.val_flip:
                        images = torch.cat([images, images.flip(-1)], dim=0)
                    masks = classify(images)

                _, prediction = masks.max(dim=1)

                if eval_pseudolabeler:
                    labels[labels < self.old_classes] = 0
                labels = labels.cpu().numpy()
                prediction = prediction.cpu().numpy()
                metrics.update(labels, prediction)

            # collect statistics from multiple processes
            metrics.synch(device)
            score = metrics.get_results()

        return score

    def load_step_ckpt(self, path):
        # generate model from path
        if osp.exists(path):
            step_checkpoint = torch.load(path, map_location="cpu")
            self.model.load_state_dict(step_checkpoint['model_state'], strict=False)  # False for incr. classifiers
            if self.opts.init_balanced:
                # implement the balanced initialization (new cls has weight of background and bias = bias_bkg - log(N+1)
                self.model.module.init_new_classifier(self.device)
            # Load state dict from the model state dict, that contains the old model parameters
            new_state = {}
            for k, v in step_checkpoint['model_state'].items():
                new_state[k[7:]] = v
            if self.opts.branch == 'none' or self.opts.branch == 'seg':
                self.model_old.load_state_dict(new_state, strict=False)
            else:
                self.model_old.load_state_dict(new_state, strict=True)  # Load also here old parameters

            self.logger.info(f"[!] Previous model loaded from {path}")
            # clean memory
            del step_checkpoint['model_state']
        elif self.opts.debug:
            self.logger.info(f"[!] WARNING: Unable to find of step {self.opts.step - 1}! "
                             f"Do you really want to do from scratch?")
        else:
            raise FileNotFoundError(path)

    def load_ckpt(self, path):
        opts = self.opts
        assert osp.isfile(path), f"Error, ckpt not found in {path}"

        checkpoint = torch.load(opts.ckpt, map_location="cpu")
        self.model.load_state_dict(checkpoint["model_state"], strict=True)
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state"])
        if "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])
        if self.weakly:
            if self.pseudolabeler is not None:
                self.pseudolabeler.load_state_dict(checkpoint["pseudolabeler"])
            if self.peakgenerator is not None:
                self.peakgenerator.load_state_dict(checkpoint["peakgenerator"])

        cur_epoch = checkpoint["epoch"] + 1
        best_score = checkpoint['best_score']
        self.logger.info("[!] Model restored from %s" % opts.ckpt)
        # if we want to resume training, resume trainer from checkpoint
        del checkpoint

        return cur_epoch, best_score
    
    def load_seg_ckpt(self, path):
        opts = self.opts
        assert osp.isfile(path), f"Error, ckpt not found in {path}"
        
        checkpoint = torch.load(opts.seg_ckpt, map_location="cpu")
        self.model.load_state_dict(checkpoint["model_state"], strict=False)
        if self.weakly:
            if self.pseudolabeler is not None:
                self.pseudolabeler.load_state_dict(checkpoint["pseudolabeler"])
                print("load pseudolabeler")
            if self.peakgenerator is not None:
                self.peakgenerator.load_state_dict(checkpoint["peakgenerator"])
                print("load peakgenerator")
        self.logger.info("[!] Seg branch restored from %s" % opts.seg_ckpt)
        
        del checkpoint
