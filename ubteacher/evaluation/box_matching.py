import torch
from detectron2.structures import Boxes, pairwise_iou


def convert_2d_idx(idx, num_r, num_c):
    r = idx // num_c
    c = idx - r * num_c
    return torch.stack([r, c], dim=-1)


def box_matching(
    gt_boxes: torch.Tensor,
    gt_classes: torch.Tensor,
    pseudo_boxes: torch.Tensor,
    pseudo_classes: torch.Tensor,
    pseudo_ious: torch.Tensor = None,
):
    results = {
        "num_gt_boxes": gt_boxes.shape[0],
        "num_pseudo_boxes": pseudo_boxes.shape[0],
        "matching_on_gt": matching_based_on_gt(
            gt_boxes, gt_classes,
            pseudo_boxes, pseudo_classes, pseudo_ious
        ),
        "matching_labels_on_gt": matching_labels(
            gt_boxes, gt_classes,
            pseudo_boxes, pseudo_classes, pseudo_ious
        ),
    }

    return results


def matching_based_on_gt(
    gt_boxes: torch.Tensor,
    gt_classes: torch.Tensor,
    pseudo_boxes: torch.Tensor,
    pseudo_classes: torch.Tensor,
    pseudo_ious: torch.Tensor = None
):
    gt_classes_bincount = torch.bincount(gt_classes)
    gt_labels_list = \
        (gt_classes_bincount != 0).nonzero(as_tuple=True)[0]

    results = dict()

    for gt_cls in gt_labels_list:
        gt_idx = (gt_classes == gt_cls).nonzero(as_tuple=True)[0]
        gt_box = Boxes(gt_boxes[gt_idx])
        num_gt_box = len(gt_box)

        pseudo_idx = (pseudo_classes == gt_cls).nonzero(as_tuple=True)[0]
        pseudo_box = Boxes(pseudo_boxes[pseudo_idx])
        if pseudo_ious is not None:
            pseudo_iou = pseudo_ious[pseudo_idx]
        num_pseudo_box = len(pseudo_box)

        if num_gt_box and num_pseudo_box:
            iou = pairwise_iou(gt_box, pseudo_box)

            sorted_iou_flat_idx = iou.flatten().argsort(descending=True)
            sorted_iou_idx_list = convert_2d_idx(
                sorted_iou_flat_idx,
                num_gt_box,
                num_pseudo_box
            )

            matched_gt_idx = []
            matched_pseudo_idx = []
            ious_of_matched_pairs = []
            for gt_idx, pseudo_idx in sorted_iou_idx_list:
                if (
                        (gt_idx not in matched_gt_idx)
                        and (pseudo_idx not in matched_pseudo_idx)
                        and iou[gt_idx, pseudo_idx] > 0.0
                ):
                    matched_gt_idx.append(gt_idx)
                    matched_pseudo_idx.append(pseudo_idx)
                    ious_of_matched_pairs.append(iou[gt_idx, pseudo_idx])

            if len(ious_of_matched_pairs):
                ious_of_matched_pairs = torch.stack(ious_of_matched_pairs)
            else:
                ious_of_matched_pairs = iou.new_tensor([])

            results_idx = gt_cls.item()
            results[results_idx] = {
                "num_gt_per_cls": num_gt_box,
                "ious_of_matched_pairs": ious_of_matched_pairs
            }
            if pseudo_ious is not None:
                if len(matched_pseudo_idx):
                    matched_pseudo_idx = torch.stack(matched_pseudo_idx)
                    matched_pseudo_ious = pseudo_iou[matched_pseudo_idx]
                else:
                    matched_pseudo_ious = pseudo_boxes.new_tensor([])
                results[results_idx]['matched_pseudo_ious'] = matched_pseudo_ious

    return results


def matching_labels(
    gt_boxes: torch.Tensor,
    gt_classes: torch.Tensor,
    pseudo_boxes: torch.Tensor,
    pseudo_classes: torch.Tensor,
    pseudo_ious: torch.Tensor = None
):
    gt_boxes = Boxes(gt_boxes)
    num_gt_boxes = len(gt_boxes)

    pseudo_boxes = Boxes(pseudo_boxes)
    num_pseudo_boxes = len(pseudo_boxes)

    results = dict()
    if num_gt_boxes and num_pseudo_boxes:
        iou = pairwise_iou(gt_boxes, pseudo_boxes)

        sorted_iou_flat_idx = iou.flatten().argsort(descending=True)
        sorted_iou_idx_list = convert_2d_idx(
            sorted_iou_flat_idx,
            num_gt_boxes,
            num_pseudo_boxes
        )

        matched_gt_idx = []
        matched_pseudo_idx = []
        ious_of_matched_pairs = []
        for gt_idx, pseudo_idx in sorted_iou_idx_list:
            if (
                    (gt_idx not in matched_gt_idx)
                    and (pseudo_idx not in matched_pseudo_idx)
                    and iou[gt_idx, pseudo_idx] > 0.0
            ):
                matched_gt_idx.append(gt_idx)
                matched_pseudo_idx.append(pseudo_idx)
                ious_of_matched_pairs.append(iou[gt_idx, pseudo_idx])

        if len(matched_gt_idx):
            ious_of_matched_pairs = torch.stack(ious_of_matched_pairs)
            matched_gt_idx = torch.stack(matched_gt_idx)
            matched_gt_classes = gt_classes[matched_gt_idx]
            matched_pseudo_idx = torch.stack(matched_pseudo_idx)
            matched_pseudo_classes = pseudo_classes[matched_pseudo_idx]
            if pseudo_ious is not None:
                matched_pseudo_ious = pseudo_ious[matched_pseudo_idx]

        else:
            ious_of_matched_pairs = gt_classes.new_tensor([])
            matched_gt_classes = gt_classes.new_tensor([])
            matched_pseudo_classes = pseudo_classes.new_tensor([])
            if pseudo_ious is not None:
                matched_pseudo_ious = pseudo_ious.new_tensor([])

        results.update({
            "ious_of_matched_pairs": ious_of_matched_pairs,
            "matched_gt_labels": matched_gt_classes,
            "matched_pseudo_labels": matched_pseudo_classes
        })
        if pseudo_ious is not None:
            results.update(dict(matched_pseudo_ious=matched_pseudo_ious))

    return results
