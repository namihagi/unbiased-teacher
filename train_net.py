#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch

from ubteacher import add_ubteacher_config, add_box_cnf_config
from ubteacher.engine.trainer import UBTeacherTrainer, BaselineTrainer

# hacky way to register
from ubteacher.modeling.meta_arch.rcnn import TwoStagePseudoLabGeneralizedRCNN
from ubteacher.modeling.proposal_generator.rpn import PseudoLabRPN
from ubteacher.modeling.roi_heads.roi_heads import StandardROIHeadsPseudoLab
import ubteacher.data.datasets.builtin

from ubteacher.modeling.meta_arch.ts_ensemble import EnsembleTSModel
from ubteacher.engine.defaults import ubteacher_default_setup
from ubteacher.evaluation.coco_evaluation_with_psedo_label import COCOEvaluatorWithPseudoLabel


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_ubteacher_config(cfg)
    add_box_cnf_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    ubteacher_default_setup(cfg)
    return cfg


def main(args):
    cfg = setup(args)
    if cfg.SEMISUPNET.Trainer == "ubteacher":
        Trainer = UBTeacherTrainer
    elif cfg.SEMISUPNET.Trainer == "baseline":
        Trainer = BaselineTrainer
    else:
        raise ValueError("Trainer Name is not found.")

    if args.eval_only:
        if cfg.SEMISUPNET.Trainer == "ubteacher":
            if cfg.SEMISUPNET.LOAD_MATCHING_PTH:
                assert cfg.SEMISUPNET.EVAL_PSEUDO_LABEL

                # build Evaluator
                output_folder = os.path.join(cfg.OUTPUT_DIR, "inference_psedo_label")
                evaluator = COCOEvaluatorWithPseudoLabel(
                    distributed=True,
                    output_dir=output_folder,
                    ckpt_iter=cfg.SEMISUPNET.EVAL_CKPT_ITERATION
                )
                res = evaluator.evaluate_from_matching_pth()
            else:
                model = Trainer.build_model(cfg)
                model_teacher = Trainer.build_model(cfg)
                ensem_ts_model = EnsembleTSModel(model_teacher, model)

                DetectionCheckpointer(
                    ensem_ts_model, save_dir=cfg.OUTPUT_DIR
                ).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
                if cfg.SEMISUPNET.EVAL_PSEUDO_LABEL:
                    res = Trainer.test_psuedo_label(
                        cfg,
                        ensem_ts_model.modelTeacher
                    )
                else:
                    res = Trainer.test(
                        cfg,
                        ensem_ts_model.modelTeacher
                    )

        else:
            model = Trainer.build_model(cfg)
            DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
                cfg.MODEL.WEIGHTS, resume=args.resume
            )
            res = Trainer.test(cfg, model)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)

    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()

    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
