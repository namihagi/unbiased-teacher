# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import itertools
import logging
import os
from collections import OrderedDict

import detectron2.utils.comm as comm
import torch
from detectron2.data import MetadataCatalog
from detectron2.evaluation import DatasetEvaluator
from detectron2.utils.file_io import PathManager
from ubteacher.evaluation.box_matching import box_matching


class COCOEvaluatorWithPseudoLabel(DatasetEvaluator):
    """
    Evaluate AR for object proposals, AP for instance detection/segmentation, AP
    for keypoint detection outputs using COCO's metrics.
    See http://cocodataset.org/#detection-eval and
    http://cocodataset.org/#keypoints-eval to understand its metrics.
    The metrics range from 0 to 100 (instead of 0 to 1), where a -1 or NaN means
    the metric cannot be computed (e.g. due to no predictions made).

    In addition to COCO, this evaluator is able to support any bounding box detection,
    instance segmentation, or keypoint detection dataset.
    """

    def __init__(
        self,
        distributed=True,
        output_dir=None,
        ckpt_iter=None,
        use_fast_impl=True,
    ):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
                It must have either the following corresponding metadata:

                    "json_file": the path to the COCO format annotation

                Or it must be in detectron2's standard dataset format
                so it can be converted to COCO format automatically.
            tasks (tuple[str]): tasks that can be evaluated under the given
                configuration. A task is one of "bbox", "segm", "keypoints".
                By default, will infer this automatically from predictions.
            distributed (True): if True, will collect results from all ranks and run evaluation
                in the main process.
                Otherwise, will only evaluate the results in the current process.
            output_dir (str): optional, an output directory to dump all
                results predicted on the dataset. The dump contains two files:

                1. "instances_predictions.pth" a file that can be loaded with `torch.load` and
                   contains all the results in the format they are produced by the model.
                2. "coco_instances_results.json" a json file in COCO's result format.
            use_fast_impl (bool): use a fast but **unofficial** implementation to compute AP.
                Although the results should be very close to the official implementation in COCO
                API, it is still recommended to compute results with the official API for use in
                papers. The faster implementation also uses more RAM.
            kpt_oks_sigmas (list[float]): The sigmas used to calculate keypoint OKS.
                See http://cocodataset.org/#keypoints-eval
                When empty, it will use the defaults in COCO.
                Otherwise it should be the same length as ROI_KEYPOINT_HEAD.NUM_KEYPOINTS.
        """
        self._logger = logging.getLogger(__name__)
        self._distributed = distributed
        assert output_dir is not None
        self._output_dir = output_dir
        assert ckpt_iter is not None
        self._ckpt_iter = ckpt_iter
        self._use_fast_impl = use_fast_impl

        self._cpu_device = torch.device("cpu")

        self._metadata = MetadataCatalog.get("coco_2017_train")

    def reset(self):
        self._matching = []

    @property
    def matching(self):
        return self._matching

    def process(self, inputs, outputs):
        for input, out_instances in zip(inputs, outputs):
            out_instances = out_instances.to(self._cpu_device)
            pseudo_boxes = out_instances.gt_boxes.tensor
            pseudo_classes = out_instances.gt_classes
            pseudo_ious = out_instances.gt_ious \
                if out_instances.has("gt_ious") else None

            gt_instances = input['instances'].to(self._cpu_device)
            gt_boxes = gt_instances.gt_boxes.tensor
            gt_classes = gt_instances.gt_classes

            matching = box_matching(
                gt_boxes, gt_classes,
                pseudo_boxes, pseudo_classes, pseudo_ious
            )
            matching.update({"image_id": input["image_id"]})

            self._matching.append(matching)

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            matching = comm.gather(self._matching, dst=0)
            matching = list(itertools.chain(*matching))

            if not comm.is_main_process():
                return {}
        else:
            matching = self._matching

        if len(matching) == 0:
            self._logger.warning("[COCOEvaluator] Did not receive valid matching.")
            return {}

        self._results = OrderedDict()
        device = self._get_device_from_matching(matching)
        self._eval_matching_cls_agnostic(matching, device, base="gt")
        self._eval_matching_cls_agnostic(matching, device, base="pseudo")
        self._eval_matching_cls_specific(matching, device, base="gt")
        self._eval_matching_cls_specific(matching, device, base="pseudo")

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            # save matching
            file_path = os.path.join(self._output_dir, f"pseudo_label_at_{self._ckpt_iter}.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(matching, f)
            # save results
            file_path = os.path.join(self._output_dir, f"results_at_{self._ckpt_iter}.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(self._results, f)

        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)

    def evaluate_from_matching_pth(self):
        if self._distributed:
            comm.synchronize()
            if not comm.is_main_process():
                return {}

        # load matching
        file_path = os.path.join(self._output_dir, f"pseudo_label_at_{self._ckpt_iter}.pth")
        self._logger.info(f"Loading matching from {file_path}...")
        matching = torch.load(file_path)

        self._results = OrderedDict()
        device = self._get_device_from_matching(matching)
        self._eval_matching_cls_agnostic(matching, device, base="gt")
        self._eval_matching_cls_agnostic(matching, device, base="pseudo")
        self._eval_matching_cls_specific(matching, device, base="gt")
        self._eval_matching_cls_specific(matching, device, base="pseudo")

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            # save results
            file_path = os.path.join(self._output_dir, f"results_at_{self._ckpt_iter}.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(self._results, f)

        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)

    def _get_device_from_matching(self, matching):
        device = None
        for result in matching:
            try:
                device = result["matching_labels_on_gt"]["ious_of_matched_pairs"].device
                break
            except:
                continue
        if device is None:
            device = self._cpu_device
        return device

    def _eval_matching_cls_agnostic(self, matching_dict, device, base="gt"):
        """
        Evaluate matching labels between pseudo-label and related gt.
        """
        self._logger.info("Evaluating matching of class label on pseudo boxes...")

        if base == "gt":
            metric = "recall"
        elif base == "pseudo":
            metric = "precision"
        else:
            raise NotImplementedError

        num_gt = 0
        iou_dist_of_matched_boxes = torch.zeros(10, dtype=torch.int64, device=device)
        iou_dist_of_matched_boxes_and_cls = \
            torch.zeros(10, dtype=torch.int64, device=device)

        for matching in matching_dict:
            num_gt_per_img = matching["num_gt_boxes"]
            num_pseudo_per_img = matching["num_pseudo_boxes"]
            if base == "gt":
                num_gt += num_gt_per_img
            elif base == "pseudo":
                num_gt += num_pseudo_per_img
            else:
                raise NotImplementedError

            if num_gt_per_img and num_pseudo_per_img:
                matching_labels_on_gt = matching[f"matching_labels_on_{base}"]
                ious_of_matched_pairs = matching_labels_on_gt["ious_of_matched_pairs"]
                matched_gt_labels = matching_labels_on_gt["matched_gt_labels"]
                matched_pseudo_labels = matching_labels_on_gt["matched_pseudo_labels"]

                for iou, gt_label, pseudo_label in zip(ious_of_matched_pairs,
                                                       matched_gt_labels,
                                                       matched_pseudo_labels):
                    idx = 9 - int(iou.item() * 10 // 1)
                    iou_dist_of_matched_boxes[idx] += 1
                    if gt_label == pseudo_label:
                        iou_dist_of_matched_boxes_and_cls[idx] += 1

        num_matched_pseudo_boxes = \
            torch.cumsum(iou_dist_of_matched_boxes, 0)
        num_matched_pseudo_boxes_and_cls = \
            torch.cumsum(iou_dist_of_matched_boxes_and_cls, 0)

        if num_gt != 0:
            recall_of_pseudo_boxes = num_matched_pseudo_boxes / num_gt
            recall_of_pseudo_boxes_and_cls = num_matched_pseudo_boxes_and_cls / num_gt
        else:
            recall_of_pseudo_boxes = \
                torch.zeros_like(iou_dist_of_matched_boxes)
            recall_of_pseudo_boxes_and_cls = \
                torch.zeros_like(iou_dist_of_matched_boxes_and_cls)

        self._results[f"matching_cls_agnostic_on_{base}"] = {
            "num_gt": num_gt,
            "iou_dist_of_matched_boxes": iou_dist_of_matched_boxes,
            "iou_dist_of_matched_boxes_and_cls": iou_dist_of_matched_boxes_and_cls,
            f"{metric}_of_pseudo_boxes": recall_of_pseudo_boxes,
            f"{metric}_of_pseudo_boxes_and_cls": recall_of_pseudo_boxes_and_cls
        }

    def _eval_matching_cls_specific(self, matching_dict, device, base="gt"):
        """
        Evaluate pseudo-label at specific class.
        """
        self._logger.info("Evaluating iou dist based on gt...")

        if base == "gt":
            metric = "recall"
        elif base == "pseudo":
            metric = "precision"
        else:
            raise NotImplementedError

        class_names = self._metadata.thing_classes
        num_classes = len(self._metadata.thing_classes)

        result = OrderedDict()
        for i in range(num_classes):
            result[i] = {
                "class_name": class_names[i],
                "num_gt_per_cls": 0,
                "iou_dist_of_matched_boxes": torch.zeros(10,
                                                         dtype=torch.int64,
                                                         device=device)
            }

        for matching in matching_dict:

            for cls_idx, matching_per_cls in matching[f"matching_on_{base}"].items():
                result[cls_idx]["num_gt_per_cls"] += matching_per_cls["num_gt_per_cls"]

                for iou in matching_per_cls["ious_of_matched_pairs"]:
                    idx = 9 - int(iou.item() * 10 // 1)
                    result[cls_idx]["iou_dist_of_matched_boxes"][idx] += 1

        for i in range(num_classes):
            num_gt_per_cls = result[i]["num_gt_per_cls"]
            iou_dist_of_matched_boxes = \
                result[i]["iou_dist_of_matched_boxes"]
            if num_gt_per_cls != 0:
                num_matched_pseudo_boxes = \
                    torch.cumsum(iou_dist_of_matched_boxes, 0)
                recall_of_pseudo_boxes = \
                    num_matched_pseudo_boxes / num_gt_per_cls
            else:
                recall_of_pseudo_boxes = \
                    torch.zeros_like(iou_dist_of_matched_boxes)

            result[i]["iou_dist_of_matched_boxes"] = iou_dist_of_matched_boxes
            result[i][f"{metric}_of_pseudo_boxes"] = recall_of_pseudo_boxes

        self._results[f"matching_cls_specific_on_{base}"] = result
