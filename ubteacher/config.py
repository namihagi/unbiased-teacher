# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from detectron2.config import CfgNode as CN


def add_ubteacher_config(cfg):
    """
    Add config for semisupnet.
    """
    _C = cfg
    _C.TEST.VAL_LOSS = True

    _C.MODEL.RPN.UNSUP_LOSS_WEIGHT = 1.0
    _C.MODEL.RPN.LOSS = "CrossEntropy"
    _C.MODEL.ROI_HEADS.LOSS = "CrossEntropy"

    _C.SOLVER.IMG_PER_BATCH_LABEL = 1
    _C.SOLVER.IMG_PER_BATCH_UNLABEL = 1
    _C.SOLVER.FACTOR_LIST = (1,)

    _C.DATASETS.TRAIN_LABEL = ("coco_2017_train",)
    _C.DATASETS.TRAIN_UNLABEL = ("coco_2017_train",)
    _C.DATASETS.CROSS_DATASET = False
    _C.TEST.EVALUATOR = "COCOeval"

    _C.SEMISUPNET = CN()

    # Output dimension of the MLP projector after `res5` block
    _C.SEMISUPNET.MLP_DIM = 128

    # Semi-supervised training
    _C.SEMISUPNET.Trainer = "ubteacher"
    _C.SEMISUPNET.BBOX_THRESHOLD = 0.7
    _C.SEMISUPNET.PSEUDO_BBOX_SAMPLE = "thresholding"
    _C.SEMISUPNET.TEACHER_UPDATE_ITER = 1
    _C.SEMISUPNET.BURN_UP_STEP = 12000
    _C.SEMISUPNET.EMA_KEEP_RATE = 0.0
    _C.SEMISUPNET.UNSUP_LOSS_WEIGHT = 4.0
    _C.SEMISUPNET.SUP_LOSS_WEIGHT = 0.5
    _C.SEMISUPNET.LOSS_WEIGHT_TYPE = "standard"

    # dataloader
    # supervision level
    _C.DATALOADER.SUP_PERCENT = 100.0  # 5 = 5% dataset as labeled set
    _C.DATALOADER.RANDOM_DATA_SEED = 0  # random seed to read data
    _C.DATALOADER.RANDOM_DATA_SEED_PATH = "dataseed/COCO_supervision.txt"

    _C.EMAMODEL = CN()
    _C.EMAMODEL.SUP_CONSIST = True


def add_box_cnf_config(cfg):
    """
    Add config for semisupnet.
    """
    _C = cfg
    _C.MODEL.ROI_HEADS.IOU_HEAD = False
    _C.MODEL.ROI_HEADS.IOU_HEAD_LOSS = "L1Loss"
    _C.MODEL.ROI_BOX_HEAD.USE_DET_SCORE = False

    _C.MODEL.ROI_BOX_HEAD.BBOX_PSUEDO_REG_LOSS_TYPE = "smooth_l1"
    _C.MODEL.ROI_BOX_HEAD.ROBUST_FUNC_C = 1.0

    _C.SEMISUPNET.IOU_FILTERING = "thresholding"
    _C.SEMISUPNET.IOU_THRESHOLD = 0.5
    _C.SEMISUPNET.CALC_PSEUDO_LOC_LOSS = False
    _C.SEMISUPNET.CALC_PSEUDO_IOU_LOSS = True
    _C.SEMISUPNET.IOU_LOSS_WEIGHT = 1.0
    _C.SEMISUPNET.PSEUDO_IOU_LOSS_WEIGHT = _C.SEMISUPNET.UNSUP_LOSS_WEIGHT
    _C.SEMISUPNET.TEACHER_REFINE_STEP = ()

    _C.SEMISUPNET.EVAL_PSEUDO_LABEL = False
    _C.SEMISUPNET.EVAL_CKPT_ITERATION = None
    _C.SEMISUPNET.LOAD_MATCHING_PTH = False
