import datetime
import logging
import time
from contextlib import ExitStack

import torch
from detectron2.structures.boxes import Boxes
from detectron2.structures.instances import Instances
from detectron2.utils.comm import get_world_size
from detectron2.utils.events import EventStorage
from detectron2.utils.logger import log_every_n_seconds


def inference_on_dataset_for_pseudo_label(
    model, data_loader, evaluator, with_iou=False,
    cur_threshold=0.7, iou_filtering="thresholding", iou_threshold=0.5
):
    num_devices = get_world_size()
    # inference data loader must have a fixed length
    total = len(data_loader)
    evaluator.reset()

    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} batches".format(total))

    # initialize time logger
    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_data_time = 0
    total_compute_time = 0
    total_eval_time = 0

    # set model to train_mode for pseudo_label
    model.train()

    with EventStorage():
        with ExitStack():
            start_data_time = time.perf_counter()

            for idx, inputs in enumerate(data_loader):
                total_data_time += time.perf_counter() - start_data_time
                if idx == num_warmup:
                    start_time = time.perf_counter()
                    total_data_time = 0
                    total_compute_time = 0
                    total_eval_time = 0

                start_compute_time = time.perf_counter()
                with torch.no_grad():
                    _, _, proposals_roih_unsup_k, _ = model(
                        inputs,
                        branch="unsup_data_weak",
                        pred_iou=with_iou
                    )

                (
                    pseudo_instances
                ) = process_pseudo_label(
                    proposals_roih_unsup_k,
                    cur_threshold,
                    pseudo_label_method="thresholding",
                    with_iou=with_iou,
                    iou_filtering=iou_filtering,
                    iou_threshold=iou_threshold
                )

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                total_compute_time += time.perf_counter() - start_compute_time

                start_eval_time = time.perf_counter()
                evaluator.process(inputs, pseudo_instances)
                total_eval_time += time.perf_counter() - start_eval_time

                iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
                data_seconds_per_iter = total_data_time / iters_after_start
                compute_seconds_per_iter = total_compute_time / iters_after_start
                eval_seconds_per_iter = total_eval_time / iters_after_start
                total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start
                if idx >= num_warmup * 2 or compute_seconds_per_iter > 5:
                    eta = datetime.timedelta(seconds=int(
                        total_seconds_per_iter * (total - idx - 1)))
                    log_every_n_seconds(
                        logging.INFO,
                        (
                            f"Inference done {idx + 1}/{total}. "
                            f"Dataloading: {data_seconds_per_iter:.4f} s / iter. "
                            f"Inference: {compute_seconds_per_iter:.4f} s / iter. "
                            f"Eval: {eval_seconds_per_iter:.4f} s / iter. "
                            f"Total: {total_seconds_per_iter:.4f} s / iter. "
                            f"ETA={eta}"
                        ),
                        n=5,
                    )
                start_data_time = time.perf_counter()

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )

    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results


def process_pseudo_label(
    proposals_rpn_unsup_k, cur_threshold, pseudo_label_method="",
    with_iou=False, iou_filtering="thresholding", iou_threshold=0.5
):
    list_instances = []
    for proposal_bbox_inst in proposals_rpn_unsup_k:
        # thresholding
        if pseudo_label_method == "thresholding":
            proposal_bbox_inst = threshold_bbox(
                proposal_bbox_inst, thres=cur_threshold,
                with_iou=with_iou, iou_filtering=iou_filtering,
                iou_thres=iou_threshold
            )
        else:
            raise ValueError("Unkown pseudo label boxes methods")
        list_instances.append(proposal_bbox_inst)
    return list_instances


def threshold_bbox(
    proposal_bbox_inst, thres=0.7,
    with_iou=False, iou_filtering="thresholding",
    iou_thres=0.5
):
    valid_map = proposal_bbox_inst.scores > thres
    if with_iou and iou_filtering == "thresholding":
        valid_map = valid_map & (proposal_bbox_inst.pred_ious > iou_thres)

    # create instances containing boxes and gt_classes
    image_shape = proposal_bbox_inst.image_size
    new_proposal_inst = Instances(image_shape)

    # create box
    new_bbox_loc = proposal_bbox_inst.pred_boxes.tensor[valid_map, :]
    new_boxes = Boxes(new_bbox_loc)

    # add boxes to instances
    new_proposal_inst.gt_boxes = new_boxes
    new_proposal_inst.gt_classes = proposal_bbox_inst.pred_classes[valid_map]
    new_proposal_inst.scores = proposal_bbox_inst.scores[valid_map]
    if with_iou:
        new_proposal_inst.gt_ious = proposal_bbox_inst.pred_ious[valid_map]

    return new_proposal_inst
