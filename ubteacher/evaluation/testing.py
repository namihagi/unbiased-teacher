import logging


def print_matching_results(results):
    """
    Print main metrics in a format similar to Detectron,
    so that they are easy to copypaste into a spreadsheet.

    Args:
        results (OrderedDict[dict]): task_name -> {metric -> score}
            unordered dict can also be logger.infoed, but in arbitrary order
    """
    logger = logging.getLogger(__name__)

    for base, metric in [["gt", "recall"], ["pseudo", "precision"]]:
        matching_cls_agnostic = results[f"matching_cls_agnostic_on_{base}"]

        iou_dist_idx = [x / 10 for x in range(10)]
        iou_dist_idx = [" > {:.1f}".format(x) for x in iou_dist_idx]

        matched_boxes = matching_cls_agnostic["iou_dist_of_matched_boxes"].numpy()
        matched_boxes = ["{:6d}".format(x) for x in matched_boxes][::-1]

        matched_boxes_and_cls = matching_cls_agnostic["iou_dist_of_matched_boxes_and_cls"].numpy()
        matched_boxes_and_cls = ["{:6d}".format(x) for x in matched_boxes_and_cls][::-1]

        recall_of_boxes = matching_cls_agnostic[f"{metric}_of_pseudo_boxes"].numpy()
        recall_of_boxes = ["{:1.4f}".format(x) for x in recall_of_boxes][::-1]

        recall_of_boxes_and_cls = matching_cls_agnostic[f"{metric}_of_pseudo_boxes_and_cls"].numpy()
        recall_of_boxes_and_cls = ["{:1.4f}".format(x) for x in recall_of_boxes_and_cls][::-1]

        logger.info(f"[{base}] iou_dist             : " + ", ".join(iou_dist_idx))
        logger.info(f"[{base}] matched_boxes        : " + ", ".join(matched_boxes))
        logger.info(f"[{base}]   {metric:<19}: " + ", ".join(recall_of_boxes))
        logger.info(f"[{base}] matched_boxes_and_cls: " + ", ".join(matched_boxes_and_cls))
        logger.info(f"[{base}]   {metric:<19}: " + ", ".join(recall_of_boxes_and_cls))

        matching_cls_specific = results[f"matching_cls_specific_on_{base}"]
        for i in range(len(matching_cls_specific)):
            detail = matching_cls_specific[i]
            class_name = detail["class_name"]

            matched_boxes = detail["iou_dist_of_matched_boxes"].numpy()
            matched_boxes = ["{:6d}".format(x) for x in matched_boxes][::-1]

            recall_of_boxes = detail[f"{metric}_of_pseudo_boxes"].numpy()
            recall_of_boxes = ["{:1.4f}".format(x) for x in recall_of_boxes][::-1]

            if i % 10 == 0:
                logger.info("iou_dist             : " + ", ".join(iou_dist_idx))
            logger.info(f"[{base}] {class_name:<21}: " + ", ".join(matched_boxes))
            logger.info(f"[{base}]   {metric:<19}: " + ", ".join(recall_of_boxes))
