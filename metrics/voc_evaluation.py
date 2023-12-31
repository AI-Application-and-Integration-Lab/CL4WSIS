from __future__ import division

from collections import defaultdict
import numpy as np
import six

from chainercv.evaluations import calc_detection_voc_ap
from chainercv.utils.mask.mask_iou import mask_iou


def eval_instance_segmentation_voc(
        n_pos, score, match, use_07_metric=False):
    """Calculate average precisions based on evaluation code of PASCAL VOC.

    This function evaluates predicted masks obtained from a dataset
    which has :math:`N` images by using average precision for each class.
    The code is based on the evaluation code used in `FCIS`_.

    .. _`FCIS`: https://arxiv.org/abs/1611.07709

    Args:
        pred_masks (iterable of numpy.ndarray): See the table below.
        pred_labels (iterable of numpy.ndarray): See the table below.
        pred_scores (iterable of numpy.ndarray): See the table below.
        gt_masks (iterable of numpy.ndarray): See the table below.
        gt_labels (iterable of numpy.ndarray): See the table below.
        iou_thresh (float): A prediction is correct if its Intersection over
            Union with the ground truth is above this value.
        use_07_metric (bool): Whether to use PASCAL VOC 2007 evaluation metric
            for calculating average precision. The default value is
            :obj:`False`.

    .. csv-table::
        :header: name, shape, dtype, format

        :obj:`pred_masks`, ":math:`[(R, H, W)]`", :obj:`bool`, --
        :obj:`pred_labels`, ":math:`[(R,)]`", :obj:`int32`, \
        ":math:`[0, \#fg\_class - 1]`"
        :obj:`pred_scores`, ":math:`[(R,)]`", :obj:`float32`, \
        --
        :obj:`gt_masks`, ":math:`[(R, H, W)]`", :obj:`bool`, --
        :obj:`gt_labels`, ":math:`[(R,)]`", :obj:`int32`, \
        ":math:`[0, \#fg\_class - 1]`"

    Returns:
        dict:

        The keys, value-types and the description of the values are listed
        below.

        * **ap** (*numpy.ndarray*): An array of average precisions. \
            The :math:`l`-th value corresponds to the average precision \
            for class :math:`l`. If class :math:`l` does not exist in \
            either :obj:`pred_labels` or :obj:`gt_labels`, the corresponding \
            value is set to :obj:`numpy.nan`.
        * **map** (*float*): The average of Average Precisions over classes.

    """

    prec, rec = calc_instance_segmentation_voc_prec_rec(n_pos, score, match)

    ap = calc_detection_voc_ap(prec, rec, use_07_metric=use_07_metric)

    return {'ap': ap, 'map': np.nanmean(ap)}


def calc_instance_segmentation_voc_prec_rec(
        n_pos, score, match):
    """Calculate precision and recall based on evaluation code of PASCAL VOC.

    This function calculates precision and recall of
    predicted masks obtained from a dataset which has :math:`N` images.
    The code is based on the evaluation code used in `FCIS`_.

    .. _`FCIS`: https://arxiv.org/abs/1611.07709

    Args:
        pred_masks (iterable of numpy.ndarray): An iterable of :math:`N`
            sets of masks. Its index corresponds to an index for the base
            dataset. Each element of :obj:`pred_masks` is an object mask
            and is an array whose shape is :math:`(R, H, W)`,
            where :math:`R` corresponds
            to the number of masks, which may vary among images.
        pred_labels (iterable of numpy.ndarray): An iterable of labels.
            Similar to :obj:`pred_masks`, its index corresponds to an
            index for the base dataset. Its length is :math:`N`.
        pred_scores (iterable of numpy.ndarray): An iterable of confidence
            scores for predicted masks. Similar to :obj:`pred_masks`,
            its index corresponds to an index for the base dataset.
            Its length is :math:`N`.
        gt_masks (iterable of numpy.ndarray): An iterable of ground truth
            masks whose length is :math:`N`. An element of :obj:`gt_masks` is
            an object mask whose shape is :math:`(R, H, W)`. Note that the
            number of masks :math:`R` in each image does not need to be
            same as the number of corresponding predicted masks.
        gt_labels (iterable of numpy.ndarray): An iterable of ground truth
            labels which are organized similarly to :obj:`gt_masks`. Its
            length is :math:`N`.
        iou_thresh (float): A prediction is correct if its Intersection over
            Union with the ground truth is above this value.

    Returns:
        tuple of two lists:
        This function returns two lists: :obj:`prec` and :obj:`rec`.

        * :obj:`prec`: A list of arrays. :obj:`prec[l]` is precision \
            for class :math:`l`. If class :math:`l` does not exist in \
            either :obj:`pred_labels` or :obj:`gt_labels`, :obj:`prec[l]` is \
            set to :obj:`None`.
        * :obj:`rec`: A list of arrays. :obj:`rec[l]` is recall \
            for class :math:`l`. If class :math:`l` that is not marked as \
            difficult does not exist in \
            :obj:`gt_labels`, :obj:`rec[l]` is \
            set to :obj:`None`.

    """

    n_fg_class = max(n_pos.keys()) + 1
    prec = [None] * n_fg_class
    rec = [None] * n_fg_class

    for l in n_pos.keys():
        score_l = np.array(score[l])
        match_l = np.array(
            match[l], dtype=np.int8)

        order = score_l.argsort()[::-1]
        match_l = match_l[order]

        tp = np.cumsum(match_l == 1)
        fp = np.cumsum(match_l == 0)

        # If an element of fp + tp is 0,
        # the corresponding element of prec[l] is nan.
        prec[l] = tp / (fp + tp)
        # If n_pos[l] is 0, rec[l] is None.
        if n_pos[l] > 0:
            rec[l] = tp / n_pos[l]

    return prec, rec
