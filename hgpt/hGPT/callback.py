import os
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback, RichProgressBar, ModelCheckpoint

class progressBar(RichProgressBar):
    def __init__(self, ):
        super().__init__()

    def get_metrics(self, trainer, model):
        # Don't show the version number
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)
        return items
    
class progressLogger(Callback):
    def __init__(self,
                 logger,
                 metric_monitor: dict,
                 precision: int = 3,
                 log_every_n_steps: int = 1):
        # Metric to monitor
        self.logger = logger
        self.metric_monitor = metric_monitor
        self.precision = precision
        self.log_every_n_steps = log_every_n_steps

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule,
                       **kwargs) -> None:
        self.logger.info("Training started")

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule,
                     **kwargs) -> None:
        self.logger.info("Training done")

    def on_validation_epoch_end(self, trainer: Trainer,
                                pl_module: LightningModule, **kwargs) -> None:
        if trainer.sanity_checking:
            self.logger.info("Sanity checking ok.")

    def on_train_epoch_end(self,
                           trainer: Trainer,
                           pl_module: LightningModule,
                           padding=False,
                           **kwargs) -> None:
        metric_format = f"{{:.{self.precision}e}}"
        line = f"Epoch {trainer.current_epoch}"
        if padding:
            line = f"{line:>{len('Epoch xxxx')}}"

        if trainer.current_epoch % self.log_every_n_steps == 0:
            metrics_str = []

            losses_dict = trainer.callback_metrics
            for metric_name, dico_name in self.metric_monitor.items():
                if dico_name in losses_dict:
                    metric = losses_dict[dico_name].item()
                    metric = metric_format.format(metric)
                    metric = f"{metric_name} {metric}"
                    metrics_str.append(metric)

            line = line + ": " + "   ".join(metrics_str)

        self.logger.info(line)


def build_callbacks(cfg, phase='test', logger=None, **kwargs):
    callbacks = []
    logger = logger

    # Rich Progress Bar
    callbacks.append(progressBar())

    # Checkpoint Callback
    if phase == 'train':
        callbacks.extend(getCheckpointCallback(cfg, logger=logger, **kwargs))
        
    return callbacks

def getCheckpointCallback(cfg, logger=None, **kwargs):
    callbacks = []
    
    # Logging
    metric_monitor = {
        "loss_total": "total/train",
        # "Train_jf": "recons/text2jfeats/train",
        # "Val_jf": "recons/text2jfeats/val",
        # "Train_rf": "recons/text2rfeats/train",
        # "Val_rf": "recons/text2rfeats/val",
        "MPJPE": "Metrics/MPJPE",
        "PAMPJPE": "Metrics/PAMPJPE",
        "ACCEL": "Metrics/ACCEL",
        # "APE root": "Metrics/APE_root",
        # "APE mean pose": "Metrics/APE_mean_pose",
        # "AVE root": "Metrics/AVE_root",
        # "AVE mean pose": "Metrics/AVE_mean_pose",
        "MM_dist": "Metrics/Matching_score",
        "gt_MM_dist": "Metrics/gt_Matching_score",
        "R_TOP_1": "Metrics/R_precision_top_1",
        "R_TOP_2": "Metrics/R_precision_top_2",
        "R_TOP_3": "Metrics/R_precision_top_3",
        "gt_R_TOP_1": "Metrics/gt_R_precision_top_1",
        "gt_R_TOP_2": "Metrics/gt_R_precision_top_2",
        "gt_R_TOP_3": "Metrics/gt_R_precision_top_3",
        "FID": "Metrics/FID",
        "gt_FID": "Metrics/gt_FID",
        "Diversity": "Metrics/Diversity",
        "gt_Diversity": "Metrics/gt_Diversity",
        # "Accuracy": "Metrics/accuracy",
    }
    
    callbacks.append(
        progressLogger(logger, metric_monitor=metric_monitor, log_every_n_steps=1))

    # Save the latest checkpoint
    checkpointParams = {
        'dirpath': os.path.join(cfg.FOLDER_EXP, "checkpoints"),
        'filename': "{epoch}",
        'monitor': "step",
        'mode': "max",
        'every_n_epochs': cfg.LOGGER.VAL_EVERY_STEPS,
        'save_top_k': 1,
        'save_last': True,
        'save_on_train_epoch_end': True
    }
    callbacks.append(ModelCheckpoint(**checkpointParams))

    # Save checkpoint every n*10 epochs
    # Found more than one stateful callback of type `ModelCheckpoint`
    # checkpointParams.update({
    #     'every_n_epochs': cfg.LOGGER.VAL_EVERY_STEPS * 10,
    #     'save_top_k': -1,
    #     'save_last': False
    # })
    # callbacks.append(ModelCheckpoint(**checkpointParams))

    # metrics = cfg.METRIC.TYPE
    # metric_monitor_map = {
    #     'TemosMetric': {
    #         'Metrics/APE_root': {
    #             'abbr': 'APEroot',
    #             'mode': 'min'
    #         },
    #     },
    #     'TM2TMetrics': {
    #         'Metrics/FID': {
    #             'abbr': 'FID',
    #             'mode': 'min'
    #         },
    #     'Metrics/R_precision_top_3': {
    #         'abbr': 'R3',
    #         'mode': 'max'
    #     }
    #     },
    #     'MRMetrics': {
    #         'Metrics/MPJPE': {
    #             'abbr': 'MPJPE',
    #             'mode': 'min'
    #         }
    #     },
    #     'HUMANACTMetrics': {
    #         'Metrics/Accuracy': {
    #             'abbr': 'Accuracy',
    #             'mode': 'max'
    #         }
    #     },
    #     'UESTCMetrics': {
    #         'Metrics/Accuracy': {
    #             'abbr': 'Accuracy',
    #             'mode': 'max'
    #         }
    #     },
    #     'UncondMetrics': {
    #         'Metrics/FID': {
    #             'abbr': 'FID',
    #             'mode': 'min'
    #         }
    #     }
    # }

    # checkpointParams.update({
    #     'every_n_epochs': cfg.LOGGER.VAL_EVERY_STEPS,
    #     'save_top_k': 1,
    # })

    # for metric in metrics:
    #     if metric in metric_monitor_map.keys():
    #         metric_monitors = metric_monitor_map[metric]

    #         # Delete R3 if training VAE
    #         if cfg.TRAIN.STAGE == 'vae' and metric == 'TM2TMetrics':
    #             del metric_monitors['Metrics/R_precision_top_3']

    #         for metric_monitor in metric_monitors:
    #             checkpointParams.update({
    #                 'filename':
    #                 metric_monitor_map[metric][metric_monitor]['mode']
    #                 + "-" +
    #                 metric_monitor_map[metric][metric_monitor]['abbr']
    #                 + "{ep}",
    #                 'monitor':
    #                 metric_monitor,
    #                 'mode':
    #                 metric_monitor_map[metric][metric_monitor]['mode'],
    #             })
    #             callbacks.append(
    #                 ModelCheckpoint(**checkpointParams))
    return callbacks