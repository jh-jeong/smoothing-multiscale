# this file is based on code publicly available at
#   https://github.com/locuslab/smoothing
# written by Jeremy Cohen.

""" This script loads a base classifier and then runs PREDICT on many examples from a dataset."""
import argparse
import os
import datetime
from time import time
import csv

import torch
from timm.models import create_model
import numpy as np
import pandas as pd

from third_party.core_mdds import Smooth
from third_party.imagenet_tools import LogitMaskingLayer
from predict_utils import ResizeLayer, get_dataset, get_diffusion_model
import models


parser = argparse.ArgumentParser(description='Predict on many examples')
parser.add_argument("dataset", type=str, help="which dataset")
parser.add_argument("base_classifier", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument("outfile", type=str, help="output file")

parser.add_argument('--sigma25', action='store_true')
parser.add_argument('--sigma50', action='store_true')
parser.add_argument('--sigma100', action='store_true')

parser.add_argument('--arch', default='CLIP_B16', type=str,
                    help='Name of model to train')
parser.add_argument('--input_size', default=224, type=int,
                    help='images input size')
parser.add_argument('--data_path', default='/data/', type=str)
parser.add_argument('--corr_type', default='all', type=str)
parser.add_argument('--ddpm_path', default=None, type=str)

parser.add_argument("--skip_p", type=float, default=0.5, help="number of samples to use")
parser.add_argument("--unif_w", type=float, default=0.1, help="number of samples to use")

parser.add_argument("--batch", type=int, default=1000, help="batch size")
parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--N", type=int, default=200, help="number of samples to use")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
parser.add_argument("--start", type=int, default=0, help='start')
args = parser.parse_args()


class DDPM(torch.nn.Module):
    def __init__(self, model, denoiser, diffusion, sigma):
        super(DDPM, self).__init__()
        self.model = model
        self.denoiser = denoiser
        self.diffusion = diffusion
        self.sigma = sigma
        self.timestep, self.sqrt_calpha = self.find_timestep(sigma, diffusion)

    def find_timestep(self, sigma, diffusion):
        schedule = diffusion.sqrt_alphas_cumprod
        sqrt_calpha = np.sqrt(1 / (1 + sigma ** 2))
        for t, sac in enumerate(schedule):
            if sac <= sqrt_calpha:
                break
        return t, sac

    def _denoise(self, x):
        x_scale = x * self.sqrt_calpha
        t = torch.tensor([self.timestep] * x.size(0), device=x.device).long()
        out = self.diffusion.p_mean_variance(self.denoiser, x_scale, t)
        x0 = out['pred_xstart']
        return x0

    def forward(self, x):
        return self.model(self._denoise(x))


def _load_model(args, n_classes):
    checkpoint = torch.load(args.base_classifier, map_location='cpu')
    base_classifier = create_model(
        args.arch, pretrained=False, num_classes=n_classes,
        drop_rate=0., drop_path_rate=0., attn_drop_rate=0., drop_block_rate=None,
        use_mean_pooling=True, init_scale=0.001,
        use_rel_pos_bias=False, use_abs_pos_emb=True, init_values=None,
    )
    base_classifier.load_state_dict(checkpoint['model_ema'])

    # clip normalization
    # mean = [0.48145466, 0.4578275, 0.40821073]
    # std = [0.26862954, 0.26130258, 0.27577711]
    resize = ResizeLayer(args.input_size)
    base_classifier = torch.nn.Sequential(resize, base_classifier)
    if args.dataset in ['imagenet_a', 'imagenet_r']:
        lmsk = LogitMaskingLayer(args.dataset)
        base_classifier = torch.nn.Sequential(base_classifier, lmsk)

    return base_classifier


def main(args, base_classifier=None):
    dataset, n_classes = get_dataset(args, args.dataset)
    if base_classifier is None:
        # load the base classifier
        base_classifier = _load_model(args, n_classes)
    base_classifier = base_classifier.cuda()

    if args.dataset in ['imagenet_a', 'imagenet_r']:
        n_classes = 200

    denoising_model, diffusion = get_diffusion_model(args.dataset, args.ddpm_path)

    # create the smoothed classifier g
    magnitude = 2.

    smooth = []
    if args.sigma25:
        ddpm25 = DDPM(base_classifier, denoising_model, diffusion, magnitude * 0.25).cuda()
        smooth25 = Smooth(ddpm25, n_classes, magnitude * 0.25)
        smooth.append(smooth25)
    if args.sigma50:
        ddpm50 = DDPM(base_classifier, denoising_model, diffusion, magnitude * 0.5).cuda()
        smooth50 = Smooth(ddpm50, n_classes, magnitude * 0.5)
        smooth.append(smooth50)
    if args.sigma100:
        ddpm100 = DDPM(base_classifier, denoising_model, diffusion, magnitude * 1.0).cuda()
        smooth100 = Smooth(ddpm100, n_classes, magnitude * 1.0)
        smooth.append(smooth100)

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    if os.path.exists(args.outfile):
        raise 'File already exists.'
    f = open(args.outfile, 'w')
    print("idx\tlabel\tpred0\tpredict\tcorrect\ttime\tstage\tc_max", file=f, flush=True)

    # iterate through the dataset
    print("Data size: ", len(dataset))
    for i in range(args.start, len(dataset)):

        # only certify every args.skip examples, and stop after args.max examples
        if i == args.max:
            break
        if i % args.skip != 0:
            continue

        (x, label) = dataset[i]
        x = x.cuda()
        base_classifier.eval()

        with torch.cuda.amp.autocast():
            before_time = time()
            logits = base_classifier(x[None])
            outputs = torch.softmax(logits, dim=1)

            c_cls = outputs.amax(1).item()
            logits = logits[0].detach().cpu().numpy()
            conf_con = None

            confidences = []
            for stage, s in enumerate(smooth[::-1]):
                pred0, conf = s.predict_threshold(x, args.N, args.skip_p, args.batch)
                if pred0 != -1:
                    radius_neg = max([-100.] + [conf.off_class_upper_radius(args.alpha, pred0) for conf in confidences])
                    conf_con = conf.counts / args.N
                    if radius_neg >= 0:
                        pred0 = -1
                        conf_con = None
                    break

            outputs = logits
            if conf_con is not None:
                prior = (1 - args.unif_w) * conf_con + args.unif_w * (1. / len(conf_con))
                log_prior = np.log(prior.clip(min=1e-20))
                outputs = logits + log_prior

            prediction = outputs.argmax()
            after_time = time()

        correct = int(prediction == label)
        time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))

        # log the prediction and whether it was correct
        print("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.4}".format(
            i, label, pred0, prediction, correct, time_elapsed, stage, c_cls), file=f, flush=True)

    f.close()

    df = pd.read_csv(args.outfile, delimiter="\t")
    acc = df['correct'].mean() * 100.
    print(f"Accuracy ({args.dataset}): {acc} %")

    return acc


if __name__ == "__main__":
    # prepare output file
    args.outdir = os.path.dirname(args.outfile)

    _dataset = args.dataset
    _accuracy = {}

    sm_str = ''
    if args.sigma25:
        sm_str += '_0.25'
    if args.sigma50:
        sm_str += '_0.50'
    if args.sigma100:
        sm_str += '_1.00'

    if args.dataset == 'cifar10c':
        from predict_utils import CORRUPTIONS, CORRUPTIONS_PER_TYPE
        args.outdir = args.outfile
        # pre-load the base classifier
        base_classifier = _load_model(args, 10)

        ctypes = args.corr_type.split(',')
        corruptions = []
        for ct in ctypes:
            if ct == 'all':
                corruptions = CORRUPTIONS
                break
            if ct not in CORRUPTIONS_PER_TYPE:
                raise NotImplementedError()
            corruptions += CORRUPTIONS_PER_TYPE[ct]

        for corruption in corruptions:
            args.outfile = f"{args.outdir}/{corruption}.tsv"
            args.dataset = f"cifar10c_{corruption}"
            acc = main(args, base_classifier)
            _accuracy[args.dataset] = acc
    else:
        acc = main(args)
        _accuracy[args.dataset] = acc

    out = f'{args.outdir}/accuracy_{_dataset}_{args.corr_type}_N{args.N}{sm_str}_P{args.skip_p}_sk{args.skip}_st{args.start}.csv'
    with open(out, 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in _accuracy.items():
            writer.writerow([key, value])
