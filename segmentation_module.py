import torch
import torch.nn as nn
import torch.nn.functional as functional

import inplace_abn
from inplace_abn import InPlaceABNSync, InPlaceABN, ABN

from functools import partial, reduce
from collections import OrderedDict

import models
from modules import DeeplabV3, custom_bn, IncrementalPanopticDeepLabDecoder, IncrementalSinglePanopticDeepLabHead
import torch.distributed as distributed
import numpy as np


def get_norm(opts):
    if opts.norm_act == 'iabn_sync':
        norm = partial(InPlaceABNSync, activation="leaky_relu", activation_param=.01, group=distributed.group.WORLD)
    elif opts.norm_act == 'iabn':
        norm = partial(InPlaceABN, activation="leaky_relu", activation_param=.01)
    elif opts.norm_act == 'abr':
        norm = partial(custom_bn.ABR, activation="leaky_relu", activation_param=.01)
    elif opts.norm_act == 'iabr':
        norm = partial(custom_bn.InPlaceABR, activation="leaky_relu", activation_param=.01)
    else:  # std bn + leaky RELU -> NO INPLACE here
        norm = partial(ABN, activation="leaky_relu", activation_param=.01)

    return norm


def get_body(opts, norm):
    if opts.model == 'PanopticDeepLab':
        body = models.__dict__[f'net_{opts.backbone}'](norm_act=norm, output_stride=opts.output_stride, keep_outputs=True)
    else:
        body = models.__dict__[f'net_{opts.backbone}'](norm_act=norm, output_stride=opts.output_stride)
    if not opts.no_pretrained:
        if opts.backbone == "wider_resnet38_a2":
            pretrained_path = f'pretrained/wide_resnet38_ipabn_lr_256.pth.tar'
        else:
            pretrained_path = f'pretrained/{opts.backbone}_iabn_sync.pth.tar'
        pre_dict = torch.load(pretrained_path, map_location='cpu')

        new_state = {}
        for k, v in pre_dict['state_dict'].items():
            if "module" in k:
                new_state[k[7:]] = v
            else:
                new_state[k] = v

        if 'classifier.fc.weight' in new_state:
            del new_state['classifier.fc.weight']
            del new_state['classifier.fc.bias']
        body.load_state_dict(new_state)
            
        del pre_dict  # free memory
        del new_state
    return body


def make_model(opts, classes=None, return_attn=False):
    norm = get_norm(opts)
    body = get_body(opts, norm)

    if opts.model == 'DeeplabV3':
        head_channels = 256
        head = DeeplabV3(body.out_channels, head_channels, 256, norm_act=norm,
            out_stride=opts.output_stride, pooling_size=opts.pooling) 

        if classes is not None:
            model = IncrementalSegmentationModule(body, head, head_channels, classes=classes)
        else:
            model = SegmentationModule(body, head, head_channels, opts.num_classes)
    elif opts.model == 'PanopticDeepLab':
        # branches
        head = None
        branch = []
        if opts.use_DeeplabV3_as_seg_branch:
            head_channels = 256
            head = DeeplabV3(body.out_channels, head_channels, 256, norm_act=norm,
                            out_stride=opts.output_stride, pooling_size=opts.pooling)
        if opts.branch == 'all':
            branch = ['seg', 'ins']
        elif opts.branch == 'seg':
            branch = ['seg']
        elif opts.branch == 'ins':
            branch = ['ins']
        
        # low_level_channels
        low_level_channels = None
        # others
        detach_instance = opts.step > 0 and opts.weakly and opts.pseudo is None or opts.detach_instance
        
        if classes is not None:
            model = IncrementalInstanceSegmentationModule(body, classes=classes, branch=branch, seg_head=head, detach_instance=detach_instance, low_level_channels=low_level_channels)

    return model


def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]


class IncrementalClassifier(nn.ModuleList):
    def forward(self, input):
        out = []
        for mod in self:
            out.append(mod(input))
        sem_logits = torch.cat(out, dim=1)
        return sem_logits


class IncrementalSegmentationModule(nn.Module):

    def __init__(self, body, head, head_channels, classes):
        super(IncrementalSegmentationModule, self).__init__()
        self.body = body
        self.head = head
        # classes must be a list where [n_class_task[i] for i in tasks]
        assert isinstance(classes, list), \
            "Classes must be a list where to every index correspond the num of classes for that task"
        self.cls = IncrementalClassifier(
            [nn.Conv2d(head_channels, c, 1) for c in classes]
        )
        self.classes = classes
        self.head_channels = head_channels
        self.tot_classes = reduce(lambda a, b: a + b, self.classes)

    def init_new_classifier(self, device):
        cls = self.cls[-1]
        imprinting_w = self.cls[0].weight[0]
        bkg_bias = self.cls[0].bias[0]

        bias_diff = torch.log(torch.FloatTensor([self.classes[-1] + 1])).to(device)

        new_bias = (bkg_bias - bias_diff)

        cls.weight.data.copy_(imprinting_w)
        cls.bias.data.copy_(new_bias)

        self.cls[0].bias[0].data.copy_(new_bias.squeeze(0))

    def forward(self, x, as_feature_extractor=False, interpolate=True, scales=None, do_flip=False):
        out_size = x.shape[-2:]

        x_b, x_b3 = self.body(x, ret_int=True)
        if not as_feature_extractor:
            x_pl = self.head(x_b)

            sem_logits = self.cls(x_pl)

            if interpolate:
                sem_logits = functional.interpolate(sem_logits, size=out_size, mode="bilinear", align_corners=False)

            return sem_logits, {"body": x_b, "pre_logits": x_pl, 'b3': x_b3}
        else:
            return {"body": x_b, 'b3': x_b3}

    def freeze(self):
        for par in self.parameters():
            par.requires_grad = False

    def fix_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, inplace_abn.ABN):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False


class _MeanFusion:
    def __init__(self, x, classes):
        self.buffer = x.new_zeros(x.size(0), classes, x.size(2), x.size(3))
        self.counter = 0

    def update(self, sem_logits):
        # probs = F.softmax(sem_logits, dim=1)
        self.counter += 1
        self.buffer.add_((sem_logits - self.buffer) / self.counter)

    def output(self):
        _, cls = self.buffer.max(1)
        return self.buffer, cls


class _SumFusion:
    def __init__(self, x, classes):
        self.buffer = x.new_zeros(x.size(0), classes, x.size(2), x.size(3))
        self.counter = 0

    def update(self, sem_logits):
        self.counter += 1
        self.buffer.add_(sem_logits)

    def output(self):
        _, cls = self.buffer.max(1)
        return self.buffer, cls


class TestAugmentation:
    def __init__(self, classes, scales=None, do_flip=True, fusion='mean'):
        self.scales = scales if scales is not None else [1.]
        self.do_flip = do_flip
        self.fusion_cls = _MeanFusion if fusion == "mean" else _SumFusion
        self.classes = classes

    def __call__(self, func, x):

        fusion = self.fusion_cls(x, self.classes)
        out_size = x.shape[-2:]

        for scale in self.scales:
            # Main orientation
            if scale != 1:
                scaled_size = [round(s * scale) for s in x.shape[-2:]]
                x_up = nn.functional.interpolate(x, size=scaled_size, mode="bilinear", align_corners=False)
            else:
                x_up = x
            # Flipped orientation
            if self.do_flip:
                x_up = torch.cat((x_up, flip(x_up, -1)), dim=0)

            sem_logits = func(x_up)
            sem_logits = nn.functional.interpolate(sem_logits, size=out_size, mode="bilinear", align_corners=False)

            if self.do_flip:
                fusion.update(flip(sem_logits[1].unsqueeze(0), -1))
                sem_logits = sem_logits[0].unsqueeze(0)

            fusion.update(sem_logits)

        return fusion.output()


class SegmentationModule(nn.Module):

    def __init__(self, body, head, head_channels, classifier):
        super(SegmentationModule, self).__init__()
        self.body = body
        self.head = head
        self.head_channels = head_channels
        self.cls = classifier

    def forward(self, x, use_classifier=True, return_feat=False, return_body=False,
                only_classifier=False, only_head=False):

        if only_classifier:
            return self.cls(x)
        elif only_head:
            return self.cls(self.head(x))
        else:
            x_b = self.body(x)
            if isinstance(x_b, dict):
                x_b = x_b["out"]
            out = self.head(x_b)

            out_size = x.shape[-2:]

            if use_classifier:
                sem_logits = self.cls(out)
                sem_logits = functional.interpolate(sem_logits, size=out_size, mode="bilinear", align_corners=False)
            else:
                sem_logits = out

            if return_feat:
                if return_body:
                    return sem_logits, out, x_b
                return sem_logits, out

            return sem_logits

    def freeze(self):
        for par in self.parameters():
            par.requires_grad = False

    def fix_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, inplace_abn.ABN):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False


# #######################################################################################

class IncrementalInstanceSegmentationModule(nn.Module):

    def __init__(self, body, classes, branch, seg_head=None, detach_instance=False, low_level_channels=None):
        super(IncrementalInstanceSegmentationModule, self).__init__()
        self.body = body

        self.branch = branch
        decoder_channels = 256
        
        self.decoder = None
        if len(branch) > 0:
            decoder = IncrementalPanopticDeepLabDecoder(in_channels=body.out_channels, 
                                            feature_key="res5", 
                                            low_level_channels=(int(body.out_channels / 2), int(body.out_channels / 4), int(body.out_channels / 8)) if low_level_channels is None else low_level_channels, 
                                            low_level_key=["res4", "res3", "res2"], 
                                            low_level_channels_project=(128, 64, 32), 
                                            decoder_channels=decoder_channels, 
                                            atrous_rates=(3, 6, 9),
                                            branch=branch,
                                            detach_instance=detach_instance
                                            )
            self.decoder = decoder
        
        # classes must be a list where [n_class_task[i] for i in tasks]
        assert isinstance(classes, list), \
            "Classes must be a list where to every index correspond the num of classes for that task"
        self.head = None
        self.cls = None
        self.semantic_head = None
        self.instance_head = None
        
        if seg_head is not None:
            head_channels = 256
            self.head = seg_head
            self.cls = IncrementalClassifier(
                [nn.Conv2d(head_channels, c, 1) for c in classes]
            )
        else:
            self.semantic_head = IncrementalSinglePanopticDeepLabHead(decoder_channels, [decoder_channels], [classes], ['seg'])
        if 'ins' in self.branch:
            instance_classes = classes.copy()
            instance_classes[0] -= 1 # minus background
            offset_classes = 2
            instance_head_kwargs = dict(
                decoder_channels=128,
                head_channels=(128, 32),
                num_classes=(instance_classes, offset_classes),
                class_key=["center", "offset"], 
            )
            self.instance_head = IncrementalSinglePanopticDeepLabHead(**instance_head_kwargs)
        self.classes = classes
        self.tot_classes = reduce(lambda a, b: a + b, self.classes)

    def init_new_classifier(self, device):
        seg_cls = self.segmentation_head['seg']['cls']
        center_cls = self.instance_head['center']['cls']

        for all_cls in [seg_cls, center_cls]:
            cls = all_cls[-1]
            imprinting_w = all_cls[0].weight[0]
            bkg_bias = all_cls[0].bias[0]

            bias_diff = torch.log(torch.FloatTensor([self.classes[-1] + 1])).to(device)

            new_bias = (bkg_bias - bias_diff)

            cls.weight.data.copy_(imprinting_w)
            cls.bias.data.copy_(new_bias)

            all_cls[0].bias[0].data.copy_(new_bias.squeeze(0))

    def forward(self, x, as_feature_extractor=False, interpolate=True, scales=None, do_flip=False):
        out_size = x.shape[-2:]

        outs = self.body(x)
        x_b, x_b3 = outs[-1], outs[2]
        # for panoptic deeplab decoder
        features = {"res1": outs[0],
                    "res2": outs[1],
                    "res3": outs[2],
                    "res4": outs[3],
                    "res5": outs[4]}
        
        if not as_feature_extractor:
            if self.decoder is not None:
                semantic_l, instance_l = self.decoder(features)

            pred = OrderedDict()
            # Semantic branch
            if self.head is not None: # deeplab v3
                x_pl = self.head(x_b)
                sem_logits = self.cls(x_pl)
                pred['seg'] = sem_logits # BxC+1x32x32
            else:
                semantic = self.semantic_head(semantic_l)
                for key in semantic.keys():
                    pred[key] = semantic[key] # BxC+1x128x128

            # Instance branch
            if 'ins' in self.branch:
                instance = self.instance_head(instance_l)
                for key in instance.keys():
                    pred[key] = instance[key] # center: BxCx128x128, offset: Bx1x128x128

            if interpolate:
                pred = self._upsample_predictions(pred, out_size)

            return pred, {"body": x_b, "features": features}
        else:
            return {"body": x_b, "features": features}
        
    def forward_seg(self, x, as_feature_extractor=False, interpolate=True, scales=None, do_flip=False):
        out_size = x.shape[-2:]

        outs = self.body(x)
        x_b, x_b3 = outs[-1], outs[2]
        # for panoptic deeplab decoder
        features = {"res1": outs[0],
                    "res2": outs[1],
                    "res3": outs[2],
                    "res4": outs[3],
                    "res5": outs[4]}
        
        if not as_feature_extractor:

            pred = OrderedDict()
            # Semantic branch
            if self.head is not None: # deeplab v3
                x_pl = self.head(x_b)
                sem_logits = self.cls(x_pl)
                pred['seg'] = sem_logits # BxC+1x32x32

            if interpolate:
                pred = self._upsample_predictions(pred, out_size)

            return pred, {"body": x_b, "features": features} 
        else:
            return {"body": x_b, "features": features}

    def _upsample_predictions(self, pred, input_shape):
        """Upsamples final prediction.
        Args:
            pred (dict): stores all output of the segmentation model.
            input_shape (tuple): spatial resolution of the desired shape.
        Returns:
            result (OrderedDict): upsampled dictionary.
        """
        result = OrderedDict()
        for key in pred.keys():
            out = functional.interpolate(pred[key], size=input_shape, mode='bilinear', align_corners=True)
            result[key] = out
        return result

    def freeze(self):
        for par in self.parameters():
            par.requires_grad = False

    def fix_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, inplace_abn.ABN):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False
