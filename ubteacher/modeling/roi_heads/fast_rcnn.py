# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Dict, List, Tuple, Union

import torch
from detectron2.layers import batched_nms, cat, nonzero_tuple
from detectron2.modeling.roi_heads.fast_rcnn import (FastRCNNOutputLayers,
                                                     _log_classification_stats)
from detectron2.structures import Boxes, Instances
from fvcore.nn import giou_loss, smooth_l1_loss
from torch import nn
from torch.nn import functional as F
from torchvision.ops import box_iou


class FastRCNNFocaltLossOutputLayers(FastRCNNOutputLayers):
    """apply FocalLoss for class score """

    def __init__(self, cfg, input_shape):
        super(FastRCNNFocaltLossOutputLayers, self).__init__(cfg, input_shape)
        self.num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES

        if cfg.MODEL.ROI_HEADS.IOU_HEAD:
            box_dim = len(self.box2box_transform.weights)
            in_features = self.bbox_pred.in_features
            out_features = self.bbox_pred.out_features
            out_features //= box_dim
            self.iou_pred = nn.Linear(in_features, out_features)
            self.iou_loss_fn = cfg.MODEL.ROI_HEADS.IOU_HEAD_LOSS

            # weight initialization
            nn.init.normal_(self.iou_pred.weight, std=0.01)
            nn.init.constant_(self.iou_pred.bias, 0)

    def forward(self, x, pred_iou=False):
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)

        scores = self.cls_score(x)
        proposal_deltas = self.bbox_pred(x)

        if pred_iou:
            ious = torch.sigmoid(self.iou_pred(x))
            return scores, proposal_deltas, ious
        else:
            return scores, proposal_deltas

    def losses(self, predictions, proposals, pred_iou=False,
               weight_on_iou=False):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were used
                to compute predictions. The fields ``proposal_boxes``, ``gt_boxes``,
                ``gt_classes`` are expected.

        Returns:
            Dict[str, Tensor]: dict of losses
        """
        if pred_iou:
            scores, proposal_deltas, ious = predictions
        else:
            scores, proposal_deltas = predictions

        # parse classification outputs
        gt_classes = (
            cat([p.gt_classes for p in proposals], dim=0) if len(proposals) else torch.empty(0)
        )
        _log_classification_stats(scores, gt_classes)

        # parse box regression outputs
        if len(proposals):
            proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)  # Nx4
            assert not proposal_boxes.requires_grad, "Proposals should not require gradients!"
            # If "gt_boxes" does not exist, the proposals must be all negative and
            # should not be included in regression loss computation.
            # Here we just use proposal_boxes as an arbitrary placeholder because its
            # value won't be used in self.box_reg_loss().
            gt_boxes = cat(
                [(p.gt_boxes if p.has("gt_boxes") else p.proposal_boxes).tensor for p in proposals],
                dim=0,
            )
            if pred_iou and weight_on_iou:
                gt_ious_list = []
                for p in proposals:
                    if p.has("gt_ious"):
                        gt_ious_list.append(p.gt_ious)
                    else:
                        proposal_boxes = p.proposal_boxes.tensor
                        n = proposal_boxes.size(0)
                        new_gt_ious = torch.ones(
                            size=(n,),
                            dtype=proposal_boxes.dtype,
                            device=proposal_boxes.device
                        )
                        gt_ious_list.append(new_gt_ious)

                gt_ious = cat(gt_ious_list, dim=0)
            else:
                gt_ious = None

        else:
            proposal_boxes = gt_boxes = torch.empty((0, 4), device=proposal_deltas.device)
            gt_ious = None

        losses = {
            "loss_cls": self.comput_focal_loss(scores, gt_classes, reduction="mean"),
            "loss_box_reg": self.box_reg_loss(
                proposal_boxes, gt_boxes, proposal_deltas, gt_classes,
                weight_on_iou=weight_on_iou, gt_ious=gt_ious
            ),
        }
        if pred_iou:
            losses["loss_iou"] = self.iou_reg_loss(
                proposal_boxes, gt_boxes, proposal_deltas, gt_classes, ious
            )

        return {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}

    def comput_focal_loss(self, scores, gt_classes, reduction="mean"):
        if gt_classes.numel() == 0 and reduction == "mean":
            return scores.sum() * 0.0

        FC_loss = FocalLoss(
            gamma=1.5,
            num_classes=self.num_classes,
        )
        total_loss = FC_loss(input=scores, target=gt_classes)
        total_loss = total_loss / gt_classes.shape[0]

        return total_loss

    def box_reg_loss(self, proposal_boxes, gt_boxes, pred_deltas, gt_classes,
                     weight_on_iou=False, gt_ious=None):
        """
        Args:
            All boxes are tensors with the same shape Rx(4 or 5).
            gt_classes is a long tensor of shape R, the gt class label of each proposal.
            R shall be the number of proposals.
        """
        if not weight_on_iou:
            return super().box_reg_loss(proposal_boxes, gt_boxes, pred_deltas, gt_classes)

        box_dim = proposal_boxes.shape[1]  # 4 or 5
        # Regression loss is only computed for foreground proposals (those matched to a GT)
        fg_inds = nonzero_tuple((gt_classes >= 0) & (gt_classes < self.num_classes))[0]
        if pred_deltas.shape[1] == box_dim:  # cls-agnostic regression
            fg_pred_deltas = pred_deltas[fg_inds]
        else:
            fg_pred_deltas = pred_deltas.view(-1, self.num_classes, box_dim)[
                fg_inds, gt_classes[fg_inds]
            ]

        if self.box_reg_loss_type == "smooth_l1":
            gt_pred_deltas = self.box2box_transform.get_deltas(
                proposal_boxes[fg_inds],
                gt_boxes[fg_inds],
            )
            loss_box_reg = smooth_l1_loss(
                fg_pred_deltas, gt_pred_deltas, self.smooth_l1_beta, reduction="none"
            )
        elif self.box_reg_loss_type == "giou":
            fg_pred_boxes = self.box2box_transform.apply_deltas(
                fg_pred_deltas, proposal_boxes[fg_inds]
            )
            loss_box_reg = giou_loss(fg_pred_boxes, gt_boxes[fg_inds], reduction="none")
        else:
            raise ValueError(f"Invalid bbox reg loss type '{self.box_reg_loss_type}'")

        if gt_ious is not None:
            fg_gt_ious = gt_ious.unsqueeze(dim=-1)[fg_inds]
            loss_box_reg = (loss_box_reg * fg_gt_ious).sum()
        else:
            loss_box_reg = loss_box_reg.sum()

        # The reg loss is normalized using the total number of regions (R), not the number
        # of foreground regions even though the box regression loss is only defined on
        # foreground regions. Why? Because doing so gives equal training influence to
        # each foreground example. To see how, consider two different minibatches:
        #  (1) Contains a single foreground region
        #  (2) Contains 100 foreground regions
        # If we normalize by the number of foreground regions, the single example in
        # minibatch (1) will be given 100 times as much influence as each foreground
        # example in minibatch (2). Normalizing by the total number of regions, R,
        # means that the single example in minibatch (1) and each of the 100 examples
        # in minibatch (2) are given equal influence.
        return loss_box_reg / max(gt_classes.numel(), 1.0)  # return 0 if empty

    def iou_reg_loss(self, proposal_boxes, gt_boxes, pred_deltas, gt_classes, pred_ious):
        box_dim = proposal_boxes.shape[1]  # 4 or 5
        # iou loss is only computed for foreground proposals like reg loss
        # (those matched to a GT)
        fg_inds = nonzero_tuple((gt_classes >= 0) & (gt_classes < self.num_classes))[0]
        if pred_deltas.shape[1] == box_dim:  # cls-agnostic regression
            fg_pred_deltas = pred_deltas[fg_inds]
            fg_pred_ious = pred_ious[fg_inds]
        else:
            fg_pred_deltas = pred_deltas.view(-1, self.num_classes, box_dim)[
                fg_inds, gt_classes[fg_inds]
            ]
            fg_pred_ious = pred_ious.view(-1, self.num_classes)[
                fg_inds, gt_classes[fg_inds]
            ]

        # get preditions in format [x1, x2, y1, y2]
        fg_pred_boxes = self.box2box_transform.apply_deltas(
            fg_pred_deltas, proposal_boxes[fg_inds]
        )
        # calculate iou between predictons and ground truth
        iou_targets = box_iou(fg_pred_boxes, gt_boxes[fg_inds])
        iou_ids = tuple(range(iou_targets.size(0)))
        iou_targets = iou_targets[iou_ids, iou_ids]

        if self.iou_loss_fn == "L1Loss":
            loss_iou_reg = smooth_l1_loss(
                fg_pred_ious, iou_targets.detach(),
                beta=0, reduction="sum"
            )
        elif self.iou_loss_fn == "BinaryCrossEntropy":
            loss_iou_reg = F.binary_cross_entropy(
                fg_pred_ious, iou_targets.detach(),
                reduction="sum"
            )
        else:
            raise NotImplementedError

        return loss_iou_reg / max(gt_classes.numel(), 1.0)

    def inference(self,
                  predictions: Tuple[torch.Tensor, ...],
                  proposals: List[Instances],
                  pred_iou: bool = False):
        if pred_iou:
            scores, proposal_deltas, ious = predictions
            pred_boxes = self.predict_boxes((scores, proposal_deltas), proposals)
            pred_scores = self.predict_probs((scores, proposal_deltas), proposals)
            pred_ious = self.predict_ious(ious, proposals)
            image_shapes = [x.image_size for x in proposals]
            return fast_rcnn_inference_with_iou(
                pred_boxes,
                pred_scores,
                pred_ious,
                image_shapes,
                self.test_score_thresh,
                self.test_nms_thresh,
                self.test_topk_per_image,
            )
        else:
            return super().inference(predictions, proposals)

    def predict_ious(
        self, ious: torch.Tensor, proposals: List[Instances]
    ):
        if not len(proposals):
            return []

        num_prop_per_image = [len(p) for p in proposals]
        return ious.split(num_prop_per_image, dim=0)


class FocalLoss(nn.Module):
    def __init__(
        self,
        weight=None,
        gamma=1.0,
        num_classes=80,
    ):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

        self.num_classes = num_classes

    def forward(self, input, target):
        # focal loss
        CE = F.cross_entropy(input, target, reduction="none")
        p = torch.exp(-CE)
        loss = (1 - p) ** self.gamma * CE
        return loss.sum()


def fast_rcnn_inference_with_iou(
    boxes: List[torch.Tensor],
    scores: List[torch.Tensor],
    ious: List[torch.Tensor],
    image_shapes: List[Tuple[int, int]],
    score_thresh: float,
    nms_thresh: float,
    topk_per_image: int,
):
    """
    Call `fast_rcnn_inference_single_image` for all images.

    Args:
        boxes (list[Tensor]): A list of Tensors of predicted class-specific or class-agnostic
            boxes for each image. Element i has shape (Ri, K * 4) if doing
            class-specific regression, or (Ri, 4) if doing class-agnostic
            regression, where Ri is the number of predicted objects for image i.
            This is compatible with the output of :meth:`FastRCNNOutputLayers.predict_boxes`.
        scores (list[Tensor]): A list of Tensors of predicted class scores for each image.
            Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
            for image i. Compatible with the output of :meth:`FastRCNNOutputLayers.predict_probs`.
        ious (list[Tensor]): A list of Tensors of predicted class-specific or class-agnostic
            ious for each image.
        image_shapes (list[tuple]): A list of (width, height) tuples for each image in the batch.
        score_thresh (float): Only return detections with a confidence score exceeding this
            threshold.
        nms_thresh (float):  The threshold to use for box non-maximum suppression. Value in [0, 1].
        topk_per_image (int): The number of top scoring detections to return. Set < 0 to return
            all detections.

    Returns:
        instances: (list[Instances]): A list of N instances, one for each image in the batch,
            that stores the topk most confidence detections.
        kept_indices: (list[Tensor]): A list of 1D tensor of length of N, each element indicates
            the corresponding boxes/scores index in [0, Ri) from the input, for image i.
    """
    result_per_image = [
        fast_rcnn_inference_single_image_with_iou(
            boxes_per_image, scores_per_image, ious_per_image, image_shape,
            score_thresh, nms_thresh, topk_per_image
        )
        for scores_per_image, boxes_per_image, ious_per_image, image_shape
        in zip(scores, boxes, ious, image_shapes)
    ]
    return [x[0] for x in result_per_image], [x[1] for x in result_per_image]


def fast_rcnn_inference_single_image_with_iou(
    boxes,
    scores,
    ious,
    image_shape: Tuple[int, int],
    score_thresh: float,
    nms_thresh: float,
    topk_per_image: int,
):
    """
    Single-image inference. Return bounding-box detection results by thresholding
    on scores and applying non-maximum suppression (NMS).

    Args:
        Same as `fast_rcnn_inference`, but with boxes, scores, and image shapes
        per image.

    Returns:
        Same as `fast_rcnn_inference`, but for only one image.
    """
    valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1)
    if not valid_mask.all():
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]
        ious = ious[valid_mask]

    scores = scores[:, :-1]
    num_bbox_reg_classes = boxes.shape[1] // 4
    # Convert to Boxes to use the `clip` function ...
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4
    ious = ious.view(-1, num_bbox_reg_classes)

    # 1. Filter results based on detection scores. It can make NMS more efficient
    #    by filtering out low-confidence detections.
    filter_mask = scores > score_thresh  # R x K
    # R' x 2. First column contains indices of the R predictions;
    # Second column contains indices of classes.
    filter_inds = filter_mask.nonzero()
    if num_bbox_reg_classes == 1:
        boxes = boxes[filter_inds[:, 0], 0]
    else:
        boxes = boxes[filter_mask]
    scores = scores[filter_mask]
    ious = ious[filter_mask]

    # 2. Apply NMS for each class independently.
    keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)
    if topk_per_image >= 0:
        keep = keep[:topk_per_image]
    (
        boxes, scores, ious, filter_inds
    ) = boxes[keep], scores[keep], ious[keep], filter_inds[keep]

    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes)
    result.scores = scores
    result.pred_ious = ious
    result.pred_classes = filter_inds[:, 1]
    return result, filter_inds[:, 0]
