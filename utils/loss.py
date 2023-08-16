import torch.nn as nn
import torch.nn.functional as F
import torch


def get_loss(loss_type):
    if loss_type == 'focal_loss':
        return FocalLoss(ignore_index=255, size_average=True)
    elif loss_type == 'cross_entropy':
        return nn.CrossEntropyLoss(ignore_index=255, reduction='mean')


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, size_average=True, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index=ignore_index
        self.size_average=size_average

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()


class HardNegativeMining(nn.Module):
    def __init__(self, perc=0.25):
        super().__init__()
        self.perc = perc

    def forward(self, loss):
        # inputs should be B, H, W
        B = loss.shape[0]
        loss = loss.reshape(B, -1)
        P = loss.shape[1]
        tk = loss.topk(dim=1, k=int(self.perc*P))
        loss = tk[0].mean()
        return loss


class SimpleNegativeMining(nn.Module):
    def __init__(self, perc=0.7):
        super().__init__()
        self.perc = perc

    def forward(self, loss):
        # inputs should be B, H, W
        B = loss.shape[0]
        loss = loss.reshape(B, -1)
        P = loss.shape[1]
        loss = - loss  # to get only most easier
        tk = loss.topk(dim=1, k=int(self.perc*P))
        loss = - tk[0].mean()
        return loss


class BCEWithLogitsLossWithIgnoreIndex(nn.Module):
    def __init__(self, reduction='mean', ignore_index=255):
        super().__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        # inputs of size B x C x H x W      
        n_cl = torch.tensor(inputs.shape[1]).to(inputs.device)
        labels_new = torch.where(targets != self.ignore_index, targets, n_cl)
        # replace ignore with numclasses + 1 (to enable one hot and then remove it)
        targets = F.one_hot(labels_new, inputs.shape[1] + 1).float().permute(0, 3, 1, 2)
        targets = targets[:, :inputs.shape[1], :, :]  # remove 255 from 1hot
        # targets is B x C x H x W so shape[1] is C
        loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        # loss has shape B x C x H x W
        loss = loss.sum(dim=1)  # sum the contributions of the classes
        if self.reduction == 'mean':
            # if targets have only zeros, we skip them
            return torch.masked_select(loss, targets.sum(dim=1) != 0).mean()
        elif self.reduction == 'sum':
            return torch.masked_select(loss, targets.sum(dim=1) != 0).sum()
        else:
            return loss * targets.sum(dim=1)


class IcarlLoss(nn.Module):
    def __init__(self, reduction='mean', ignore_index=255, bkg=1.):
        super().__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.bkg = bkg

    def forward(self, inputs, targets, output_old):
        # inputs of size B x C x H x W
        n_cl = torch.tensor(inputs.shape[1]).to(inputs.device)
        labels_new = torch.where(targets != self.ignore_index, targets, n_cl)
        # replace ignore with numclasses + 1 (to enable one hot and then remove it)
        targets = F.one_hot(labels_new, inputs.shape[1] + 1).float().permute(0, 3, 1, 2)
        targets = targets[:, :inputs.shape[1], :, :]  # remove 255 from 1hot

        targets[:, 1:output_old.shape[1], :, :] = output_old[:, 1:, :, :]
        if self.bkg != -1:
            targets[:, 0, :, :] = self.bkg * targets[:, 0, :, :] + (1-self.bkg)*output_old[:, 0, :, :]
        else:
            targets[:, 0, :, :] = torch.min(targets[:, 0, :, :], output_old[:, 0, :, :])

        # targets is B x C x H x W so shape[1] is C
        loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        # loss has shape B x C x H x W
        loss = loss.sum(dim=1)  # sum the contributions of the classes
        if self.reduction == 'mean':
            # if targets have only zeros, we skip them
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class UnbiasedCrossEntropy(nn.Module):
    def __init__(self, old_cl=None, reduction='mean', ignore_index=255):
        super().__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.old_cl = old_cl

    def forward(self, inputs, targets):

        old_cl = self.old_cl
        outputs = torch.zeros_like(inputs)  # B, C (1+V+N), H, W
        den = torch.logsumexp(inputs, dim=1)                               # B, H, W       den of softmax
        outputs[:, 0] = torch.logsumexp(inputs[:, 0:old_cl], dim=1) - den  # B, H, W       p(O)
        outputs[:, old_cl:] = inputs[:, old_cl:] - den.unsqueeze(dim=1)    # B, N, H, W    p(N_i)

        labels = targets.clone()    # B, H, W
        labels[targets < old_cl] = 0  # just to be sure that all labels old belongs to zero

        loss = F.nll_loss(outputs, labels, ignore_index=self.ignore_index, reduction=self.reduction)

        return loss


class KnowledgeDistillationLoss(nn.Module):
    def __init__(self, reduction='mean', alpha=1.):
        super().__init__()
        self.reduction = reduction
        self.alpha = alpha

    def forward(self, inputs, targets, mask=None):
        inputs = inputs.narrow(1, 0, targets.shape[1])

        outputs = torch.log_softmax(inputs, dim=1)
        labels = torch.softmax(targets * self.alpha, dim=1)

        loss = (outputs * labels).mean(dim=1)

        if mask is not None:
            loss = loss * mask.float()

        if self.reduction == 'mean':
            outputs = -torch.mean(loss)
        elif self.reduction == 'sum':
            outputs = -torch.sum(loss)
        else:
            outputs = -loss

        return outputs


class UnbiasedKnowledgeDistillationLoss(nn.Module):
    def __init__(self, reduction='mean', alpha=1.):
        super().__init__()
        self.reduction = reduction
        self.alpha = alpha

    def forward(self, inputs, targets, mask=None):

        new_cl = inputs.shape[1] - targets.shape[1]

        targets = targets * self.alpha

        new_bkg_idx = torch.tensor([0] + [x for x in range(targets.shape[1], inputs.shape[1])]).to(inputs.device)

        den = torch.logsumexp(inputs, dim=1)                          # B, H, W
        outputs_no_bgk = inputs[:, 1:-new_cl] - den.unsqueeze(dim=1)  # B, OLD_CL, H, W
        outputs_bkg = torch.logsumexp(torch.index_select(inputs, index=new_bkg_idx, dim=1), dim=1) - den     # B, H, W

        labels = torch.softmax(targets, dim=1)                        # B, BKG + OLD_CL, H, W

        # make the average on the classes 1/n_cl \sum{c=1..n_cl} L_c
        loss = (labels[:, 0] * outputs_bkg + (labels[:, 1:] * outputs_no_bgk).sum(dim=1)) / targets.shape[1]

        if mask is not None:
            loss = loss * mask.float()

        if self.reduction == 'mean':
                outputs = -torch.mean(loss)
        elif self.reduction == 'sum':
                outputs = -torch.sum(loss)
        else:
            outputs = -loss

        return outputs

"""
BESTIE
Copyright (c) 2022-present NAVER Corp.
MIT License
"""

import torch

    
class L1_Loss(torch.nn.Module):
    ''' L1 loss for Offset map (without Instance-aware Guidance)'''
    def __init__(self):
        super(L1_Loss, self).__init__()
        self.l1_loss = torch.nn.L1Loss(reduction='mean')

    def forward(self, out, target, weight):
        loss = self.l1_loss(out, target)
        
        return loss
    
    
class Weighted_L1_Loss(torch.nn.Module):
    ''' Weighted L1 loss for Offset map (with Instance-aware Guidance)'''
    def __init__(self):
        super(Weighted_L1_Loss, self).__init__()
        self.l1_loss = torch.nn.L1Loss(reduction='none')

    def forward(self, out, target, weight):
        loss = self.l1_loss(out, target) * weight
        
        assert (weight >= 0).all()
        if weight.sum() > 0:
            loss = loss.sum() / (weight > 0).float().sum()
        else:
            loss = loss.sum() * 0
        
        return loss
    
    
class MSELoss(torch.nn.Module):
    ''' MSE loss for center map (without Instance-aware Guidance)'''
    def __init__(self):
        super(MSELoss, self).__init__()
        self.mse_loss = torch.nn.MSELoss(reduction='mean')

    def forward(self, out, target, weight):
        
        loss = self.mse_loss(out, target)
        
        return loss
        

class Weighted_MSELoss(torch.nn.Module):
    ''' MSE loss for center map (with Instance-aware Guidance)'''
    def __init__(self):
        super(Weighted_MSELoss, self).__init__()
        self.mse_loss = torch.nn.MSELoss(reduction='none')

    def forward(self, out, target, weight):
        
        loss = self.mse_loss(out, target) * weight
        
        assert (weight >= 0).all()
        if weight.sum() > 0:
            loss = loss.sum() / (weight > 0).float().sum()
        else:
            loss = loss.sum() * 0
        
        return loss

    
class DeepLabCE(torch.nn.Module):
    """
    Hard pixel mining mining with cross entropy loss, for semantic segmentation.
    This is used in TensorFlow DeepLab frameworks.
    Reference: https://github.com/tensorflow/models/blob/bd488858d610e44df69da6f89277e9de8a03722c/research/deeplab/utils/train_utils.py#L33
    Arguments:
        ignore_label: Integer, label to ignore.
        top_k_percent_pixels: Float, the value lies in [0.0, 1.0]. When its value < 1.0, only compute the loss for
            the top k percent pixels (e.g., the top 20% pixels). This is useful for hard pixel mining.
        weight: Tensor, a manual rescaling weight given to each class.
    """
    def __init__(self, ignore_label=255, top_k_percent_pixels=0.2, weight=None):
        super(DeepLabCE, self).__init__()
        self.top_k_percent_pixels = top_k_percent_pixels
        self.ignore_label = ignore_label
        self.criterion = torch.nn.CrossEntropyLoss(weight=weight,
                                             ignore_index=ignore_label,
                                             reduction='none')

    def forward(self, logits, labels):
        
        pixel_losses = self.criterion(logits, labels).contiguous().view(-1)
        
        if self.top_k_percent_pixels == 1.0:
            return pixel_losses.mean()

        top_k_pixels = int(self.top_k_percent_pixels * pixel_losses.numel())
        pixel_losses, _ = torch.topk(pixel_losses, top_k_pixels)
        
        return pixel_losses.mean()

class DeepLabBCE(torch.nn.Module):
    """
    Hard pixel mining mining with cross entropy loss, for semantic segmentation.
    This is used in TensorFlow DeepLab frameworks.
    Reference: https://github.com/tensorflow/models/blob/bd488858d610e44df69da6f89277e9de8a03722c/research/deeplab/utils/train_utils.py#L33
    Arguments:
        ignore_label: Integer, label to ignore.
        top_k_percent_pixels: Float, the value lies in [0.0, 1.0]. When its value < 1.0, only compute the loss for
            the top k percent pixels (e.g., the top 20% pixels). This is useful for hard pixel mining.
        weight: Tensor, a manual rescaling weight given to each class.
    """
    def __init__(self, top_k_percent_pixels=0.2, weight=None):
        super(DeepLabBCE, self).__init__()
        self.top_k_percent_pixels = top_k_percent_pixels
        self.criterion = torch.nn.BCEWithLogitsLoss(weight=weight,
                                                    reduction='none')

    def forward(self, logits, labels, weight):
        
        pixel_losses = self.criterion(logits, labels) * weight

        pixel_losses = pixel_losses.contiguous().view(-1)
        
        if self.top_k_percent_pixels == 1.0:
            # if weight.sum() > 0:
            #     pixel_losses = pixel_losses / (weight > 0).float()
            # else:
            #     pixel_losses = pixel_losses * 0
            
            return pixel_losses.mean()

        top_k_pixels = int(self.top_k_percent_pixels * pixel_losses.numel())
        pixel_losses, _ = torch.topk(pixel_losses, top_k_pixels)
        
        # if weight.sum() > 0:
        #     pixel_losses = pixel_losses / (weight > 0).float()
        # else:
        #     pixel_losses = pixel_losses * 0
        
        return pixel_losses.mean()
    
class RegularCE(torch.nn.Module):
    """
    Regular cross entropy loss for semantic segmentation, support pixel-wise loss weight.
    Arguments:
        ignore_label: Integer, label to ignore.
        weight: Tensor, a manual rescaling weight given to each class.
    """
    def __init__(self, ignore_label=255, weight=None):
        super(RegularCE, self).__init__()
        self.ignore_label = ignore_label
        self.criterion = torch.nn.CrossEntropyLoss(weight=weight,
                                             ignore_index=ignore_label,
                                             reduction='none')

    def forward(self, logits, labels):
        pixel_losses = self.criterion(logits, labels)
        
        mask = (labels != self.ignore_label)

        if mask.sum() > 0:
            pixel_losses = pixel_losses.sum() / mask.sum()
        else:
            pixel_losses = pixel_losses.sum() * 0
        
        return pixel_losses

    
    
def _neg_loss(pred, gt, weight):
    ''' Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
    Arguments:
      pred (batch x c x h x w)
      gt_regr (batch x c x h x w)
    '''
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds * weight
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds * weight

    num_pos  = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


class FocalLoss(torch.nn.Module):
    '''nn.Module warpper for focal loss'''
    def __init__(self):
        super(FocalLoss, self).__init__()
        self.neg_loss = _neg_loss

    def forward(self, out, target, weight):
        return self.neg_loss(out, target, weight)
