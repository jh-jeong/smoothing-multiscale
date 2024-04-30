import torch
import torch.nn as nn
from torch.nn import functional as F

try:
    import torch.distributed.nn
    from torch import distributed as dist
    has_distributed = True
except ImportError:
    has_distributed = False


def gather_features(image_features,
                    text_features,
                    local_loss=False,
                    gather_with_grad=False,
                    rank=0,
                    world_size=1):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    # We gather tensors from all gpus
    if gather_with_grad:
        all_image_features = torch.cat(
            torch.distributed.nn.all_gather(image_features), dim=0)
        all_text_features = torch.cat(
            torch.distributed.nn.all_gather(text_features), dim=0)
    else:
        gathered_image_features = [
            torch.zeros_like(image_features) for _ in range(world_size)
        ]
        gathered_text_features = [
            torch.zeros_like(text_features) for _ in range(world_size)
        ]
        dist.all_gather(gathered_image_features, image_features)
        dist.all_gather(gathered_text_features, text_features)
        if not local_loss:
            # ensure grads for local rank when all_* features don't have a gradient
            gathered_image_features[rank] = image_features
            gathered_text_features[rank] = text_features
        all_image_features = torch.cat(gathered_image_features, dim=0)
        all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features


class CLIPLoss(nn.Module):
    def __init__(self, local_loss=False, gather_with_grad=False, cache_labels=False, rank=0, world_size=1,
                 text_embedding: str = ''):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

        text_features = torch.load(text_embedding, map_location='cpu')
        self.text_features = nn.Embedding(text_features.size(1), text_features.size(0))
        self.text_features.weight.data.copy_(text_features)
        print(f"*** Text embedding loaded @ CLIP from {text_embedding} ***")

    def forward(self, image_features, targets, logit_scale=100.):
        device = image_features.device
        text_features = self.text_features(targets).to(image_features.dtype)
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features, self.local_loss,
                self.gather_with_grad, self.rank, self.world_size)

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            # import pdb;pdb.set_trace()
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T

        # calculated ground-truth and cache if enabled
        num_logits = logits_per_image.shape[0]

        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits,
                                  device=device,
                                  dtype=torch.long)

            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]

        total_loss = (F.cross_entropy(logits_per_image, labels) +
                      F.cross_entropy(logits_per_text, labels)) / 2
        return total_loss


class DenoisingCLIPLoss(nn.Module):
    def __init__(self, logit_scale=10.,
                 local_loss=False, gather_with_grad=False, cache_labels=False, rank=0, world_size=1):
        super().__init__()
        self.logit_scale = logit_scale
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def forward(self, outputs, noisy_features, clean_features, targets, lbd=1.0):
        device = noisy_features.device
        clean_features = clean_features.detach()
        if self.world_size > 1:
            all_noisy_features, all_clean_features = gather_features(
                noisy_features, clean_features, self.local_loss,
                self.gather_with_grad, self.rank, self.world_size)

            if self.local_loss:
                logits_per_noisy = self.logit_scale * noisy_features @ all_clean_features.T
                # logits_per_clean = logit_scale * clean_features @ all_noisy_features.T
            else:
                logits_per_noisy = self.logit_scale * all_noisy_features @ all_clean_features.T
                # logits_per_clean = logits_per_noisy.T
        else:
            # import pdb;pdb.set_trace()
            logits_per_noisy = self.logit_scale * noisy_features @ clean_features.T
            # logits_per_clean = logit_scale * clean_features @ noisy_features.T

        # calculated ground-truth and cache if enabled
        num_logits = logits_per_noisy.shape[0]

        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]

        # total_loss = (F.cross_entropy(logits_per_noisy, labels) +
        #               F.cross_entropy(logits_per_clean, labels)) / 2
        denoising_loss = F.cross_entropy(logits_per_noisy, labels)
        xent_loss = F.cross_entropy(outputs, targets)
        total_loss = denoising_loss + lbd * xent_loss

        return total_loss


class GaussianCLIPLoss(nn.Module):
    def __init__(self, logit_scale=10.,
                 local_loss=False, gather_with_grad=False, cache_labels=False, rank=0, world_size=1):
        super().__init__()
        self.logit_scale = logit_scale
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def forward(self, outputs, view1, view2, targets, lbd=1.0):
        device = view1.device
        if self.world_size > 1:
            all_view1, all_view2 = gather_features(
                view1, view2, self.local_loss,
                self.gather_with_grad, self.rank, self.world_size)

            if self.local_loss:
                logits1 = self.logit_scale * view1 @ all_view2.T
                logits2 = self.logit_scale * view2 @ all_view1.T
            else:
                logits1 = self.logit_scale * all_view1 @ all_view2.T
                logits2 = logits1.T
        else:
            # import pdb;pdb.set_trace()
            logits1 = self.logit_scale * view1 @ view2.T
            logits2 = self.logit_scale * view2 @ view1.T

        # calculated ground-truth and cache if enabled
        num_logits = logits1.shape[0]

        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]

        denoising_loss = (F.cross_entropy(logits1, labels) + F.cross_entropy(logits2, labels)) / 2
        xent_loss = F.cross_entropy(outputs, targets)
        total_loss = xent_loss + lbd * denoising_loss

        return total_loss
