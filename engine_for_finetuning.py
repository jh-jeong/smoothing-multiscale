import math
import sys
from typing import Iterable, Optional

import torch
import torch.nn.functional as F

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

import utils
from consistency import consistency_loss
from third_party.clip_loss import CLIPLoss, DenoisingCLIPLoss, GaussianCLIPLoss


def train_class_batch(model, samples, target, criterion):
    if isinstance(criterion, CLIPLoss):
        outputs = model(samples, return_feature=True)
        loss = criterion(outputs, target)
    elif isinstance(criterion, DenoisingCLIPLoss):
        outputs, features = model(samples, return_feature=True, return_output=True)
        noisy_features, clean_features = features.chunk(2, dim=0)
        _, outputs = outputs.chunk(2, dim=0)
        loss = criterion(outputs, noisy_features, clean_features, target)
    elif isinstance(criterion, GaussianCLIPLoss):
        outputs, features = model(samples, return_feature=True, return_output=True)
        view1, view2 = features.chunk(2, dim=0)
        loss = criterion(outputs, view1, view2, target)
    else:
        outputs = model(samples)
        loss = criterion(outputs, target)
    return loss, outputs


def get_loss_scale_for_deepspeed(model):
    optimizer = model.optimizer
    return optimizer.loss_scale if hasattr(optimizer, "loss_scale") else optimizer.cur_scale


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, model_emas = None,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None,
                    post_transform=None, consistency_lbd: float = 0., con_loss='default',
                    jsd_lbd: float = 0., denoise=None):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    print_freq = 100
    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if denoise is not None:
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    samples = denoise(samples)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if consistency_lbd > 0:
            samples = torch.cat([samples, samples])
            targets = torch.cat([targets, targets])
        if jsd_lbd > 0:
            samples0 = samples
            samples = torch.cat([samples, samples])
        if isinstance(criterion, DenoisingCLIPLoss):
            samples0 = samples
        if isinstance(criterion, GaussianCLIPLoss):
            samples = torch.cat([samples, samples])
            targets = torch.cat([targets, targets])

        with torch.no_grad():
            if post_transform is not None:
                samples = post_transform(samples)

            if isinstance(criterion, DenoisingCLIPLoss):
                for t in post_transform.transforms[1:]:  # Ignore Gaussian noise
                    samples0 = t(samples0)
                samples = torch.cat([samples, samples0])
            if jsd_lbd > 0:
                for t in post_transform.transforms[1:]: # Ignore Gaussian noise
                    samples0 = t(samples0)
                samples1, samples2 = samples.chunk(2, dim=0)
                lam1 = torch.rand(samples0.size(0), device=device)
                lam2 = torch.rand(samples0.size(0), device=device)
                samples1 = (1 - lam1).view(-1, 1, 1, 1) * samples0 + lam1.view(-1, 1, 1, 1) * samples1
                samples2 = (1 - lam2).view(-1, 1, 1, 1) * samples0 + lam2.view(-1, 1, 1, 1) * samples2
                samples = samples0

        if loss_scaler is None:
            samples = samples.half()
            loss, output = train_class_batch(
                model, samples, targets, criterion)
        else:
            with torch.cuda.amp.autocast():
                loss, output = train_class_batch(
                    model, samples, targets, criterion)

        if consistency_lbd > 0:
            logits = torch.chunk(output, 2, dim=0)
            loss_con = consistency_loss(logits, lbd=consistency_lbd, loss=con_loss)
            loss = loss + loss_con
        if jsd_lbd > 0:
            with torch.cuda.amp.autocast():
                output1 = model(samples1)
                output2 = model(samples2)
            p_clean, p_aug1, p_aug2 = torch.softmax(output, dim=1), \
                                      torch.softmax(output1, dim=1), \
                                      torch.softmax(output2, dim=1)

            # Clamp mixture distribution to avoid exploding KL divergence
            p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
            loss_jsd = jsd_lbd * (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                                  F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
                                  F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.
            loss = loss + loss_jsd

        loss_value = loss.item()

        restart = torch.zeros(1).cuda()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value), force=True)
            restart = torch.ones(1).cuda()
            #sys.exit(1)
        restarts = torch.sum(utils.all_gather_batch([restart])[0])
        if restarts > 0:
            sys.exit(1)

        if loss_scaler is None:
            loss /= update_freq
            model.backward(loss)
            model.step()

            if (data_iter_step + 1) % update_freq == 0:
                # model.zero_grad()
                # Deepspeed will call step() & model.zero_grad() automatic
                if model_ema is not None:
                    model_ema.update(model)
                if model_emas is not None:
                    for model_ema in model_emas:
                        model_ema.update(model)
            grad_norm = None
            loss_scale_value = get_loss_scale_for_deepspeed(model)
        else:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
                if model_emas is not None:
                    for model_ema in model_emas:
                        model_ema.update(model)

            loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        if mixup_fn is None:
            class_acc = (output.max(-1)[-1] == targets).float().mean()
        else:
            class_acc = None
        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 200, header):
        images = batch[0]
        target = batch[-1]
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
