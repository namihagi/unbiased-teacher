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

    matching_cls_agnostic = results["matching_cls_agnostic"]
    matching_cls_specific = results["matching_cls_specific"]

    iou_dist_idx = [x / 10 for x in range(10)]
    iou_dist_idx = [" > {:.1f}".format(x) for x in iou_dist_idx]

    matched_boxes = matching_cls_agnostic["iou_dist_of_matched_boxes"].numpy()
    matched_boxes = ["{:6d}".format(x) for x in matched_boxes][::-1]

    matched_boxes_and_cls = matching_cls_agnostic["iou_dist_of_matched_boxes_and_cls"].numpy()
    matched_boxes_and_cls = ["{:6d}".format(x) for x in matched_boxes_and_cls][::-1]

    recall_of_boxes = matching_cls_agnostic["recall_of_pseudo_boxes"].numpy()
    recall_of_boxes = ["{:1.4f}".format(x) for x in recall_of_boxes][::-1]

    recall_of_boxes_and_cls = matching_cls_agnostic["recall_of_pseudo_boxes_and_cls"].numpy()
    recall_of_boxes_and_cls = ["{:1.4f}".format(x) for x in recall_of_boxes_and_cls][::-1]

    logger.info("iou_dist             : " + ", ".join(iou_dist_idx))
    logger.info("matched_boxes        : " + ", ".join(matched_boxes))
    logger.info("(recall)             : " + ", ".join(recall_of_boxes))
    logger.info("matched_boxes_and_cls: " + ", ".join(matched_boxes_and_cls))
    logger.info("(recall)             : " + ", ".join(recall_of_boxes_and_cls))

    for i in range(len(matching_cls_specific)):
        detail = matching_cls_specific[i]
        class_name = detail["class_name"]

        matched_boxes = detail["iou_dist_of_matched_boxes"].numpy()
        matched_boxes = ["{:6d}".format(x) for x in matched_boxes][::-1]

        recall_of_boxes = detail["recall_of_pseudo_boxes"].numpy()
        recall_of_boxes = ["{:1.4f}".format(x) for x in recall_of_boxes][::-1]

        if i % 10 == 0:
            logger.info("iou_dist             : " + ", ".join(iou_dist_idx))
        logger.info("{:<21}: ".format(class_name) + ", ".join(matched_boxes))
        logger.info("(recall)             : " + ", ".join(recall_of_boxes))
