from os.path import join as pjoin
from torch import Tensor, nn

from .mr import MRMetrics
from .t2m import TM2TMetrics
from .mm import MMMetrics
from .m2t import M2TMetrics
from .m2m import PredMetrics

class BaseMetrics(nn.Module):
    def __init__(self, cfg, datamodule, debug,
                 **kwargs) -> None:
        super().__init__()
        data_name = datamodule.name
        njoints = datamodule.njoints
        
        assert len(cfg.METRIC.TYPE) > 0
        if 'MRMetrics' in cfg.METRIC.TYPE:
            self.MRMetrics = MRMetrics(
                njoints=njoints,
                dataname=data_name,
                dist_sync_on_step=cfg.METRIC.DIST_SYNC_ON_STEP,
            )
        if 'TM2TMetrics' in cfg.METRIC.TYPE:
            self.TM2TMetrics = TM2TMetrics(
                cfg=cfg,
                dataname=data_name,
                diversity_times=30 if debug else cfg.METRIC.DIVERSITY_TIMES,
                dist_sync_on_step=cfg.METRIC.DIST_SYNC_ON_STEP,
                unit_len=datamodule.unit_len
            )
        if 'MMMetrics' in cfg.METRIC.TYPE:
            self.MMMetrics = MMMetrics(
                cfg=cfg,
                dataname=data_name,
                mm_num_times=cfg.METRIC.MM_NUM_TIMES,
                dist_sync_on_step=cfg.METRIC.DIST_SYNC_ON_STEP,
            )
        if 'M2TMetrics' in cfg.METRIC.TYPE:
            self.M2TMetrics = M2TMetrics(
                cfg=cfg,
                dataname=data_name,
                w_vectorizer=datamodule.hparams.w_vectorizer,
                diversity_times=30 if debug else cfg.METRIC.DIVERSITY_TIMES,
                dist_sync_on_step=cfg.METRIC.DIST_SYNC_ON_STEP)
        if 'PredMetrics' in cfg.METRIC.TYPE:
            self.PredMetrics = PredMetrics(
                cfg=cfg,
                dataname=data_name,
                njoints=njoints,
                dist_sync_on_step=cfg.METRIC.DIST_SYNC_ON_STEP,
                task=cfg.model.params.task,
            )