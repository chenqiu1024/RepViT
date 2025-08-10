"""
Train and eval functions used in main.py

This training loop supports knowledge distillation following DeiT-style losses
and EMA tracking from timm.

Background and relations to docs/ papers:
- RepViT and RepViT-SAM emphasize deployment efficiency; this engine uses BN
  eval freezing (set_bn_eval) and gradient clipping modes conducive to stable
  training before fusing at deploy.
- DistillationLoss mirrors DeiT (not in docs/ here) but is used in RepViT
  training to improve accuracy of small backbones; see RepViT paper for
  training recipe references.

Note: Only comments added.
"""
import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from losses import DistillationLoss
import utils

def set_bn_state(model):
    """Optionally set BatchNorm modules to eval mode during fine-tuning to
    stabilize statistics, as suggested in efficient CNN deployment literature.
    Useful prior to Conv-BN fusion used in RepViT deployment.
    """
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()

def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    clip_grad: float = 0,
                    clip_mode: str = 'norm',
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True,
                    set_bn_eval=False,):
    """Standard one-epoch loop with AMP and optional knowledge distillation.

    - Mixup/Cutmix follow timm recipes commonly used in lightweight backbones.
    - Loss is wrapped by DistillationLoss (see losses.py) to combine base loss
      and teacher supervision, in line with RepViT training setups.
    """
    model.train(set_training_mode)
    if set_bn_eval:
        set_bn_state(model)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(
        window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    for samples, targets in metric_logger.log_every(
            data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(samples, outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(
            optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=clip_grad, clip_mode=clip_mode,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    """Validation loop computing Top-1/Top-5 accuracy and cross-entropy loss."""
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
