from detectron2.engine.defaults import _try_get_key
from detectron2.utils import comm
from detectron2.utils.logger import setup_logger


def ubteacher_default_setup(cfg):
    """
    Perform some basic common setups at the beginning of a job, including:

    1. Set up the detectron2 logger
    2. Log basic information about environment, cmdline arguments, and config
    3. Backup the config to the output directory

    Args:
        cfg (CfgNode or omegaconf.DictConfig): the full config to be used
        args (argparse.NameSpace): the command line arguments to be logged
    """
    output_dir = _try_get_key(cfg, "OUTPUT_DIR", "output_dir", "train.output_dir")

    rank = comm.get_rank()
    setup_logger(output_dir, distributed_rank=rank, name="ubteacher")
