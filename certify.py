# this file is based on code publicly available at
#   https://github.com/locuslab/smoothing
# written by Jeremy Cohen.

""" Evaluate a smoothed classifier on a dataset. """
import argparse
import os
import datetime
from time import time

import torch
from timm.models import create_model
import numpy as np

from third_party.core_mdds import Smooth
from predict_utils import ResizeLayer, get_dataset, get_diffusion_model
import models


parser = argparse.ArgumentParser(description='Certify many examples')
parser.add_argument("dataset", type=str, help="which dataset")
parser.add_argument("base_classifier", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument("outfile", type=str, help="output file")

parser.add_argument('--sigma25', action='store_true')
parser.add_argument('--sigma50', action='store_true')
parser.add_argument('--sigma100', action='store_true')

parser.add_argument('--arch', default='CLIP_B16', type=str, metavar='ARCH',
                    help='Name of model to train')
parser.add_argument('--input_size', default=224, type=int,
                    help='images input size')
parser.add_argument('--data_path', default='/data/', type=str)
parser.add_argument('--ddpm_path', default=None, type=str)

parser.add_argument("--batch", type=int, default=1000, help="batch size")
parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--N0", type=int, default=100)
parser.add_argument("--N", type=int, default=100000, help="number of samples to use")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
parser.add_argument("--skip_p", type=float, default=0.5)
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


if __name__ == "__main__":
    # load the base classifier
    checkpoint = torch.load(args.base_classifier, map_location='cpu')

    dataset, n_classes = get_dataset(args, args.dataset)
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
    base_classifier = torch.nn.Sequential(resize, base_classifier).cuda()
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

    # prepare output file
    outdir = os.path.dirname(args.outfile)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    if os.path.exists(args.outfile):
        raise 'File already exists.'
    f = open(args.outfile, 'w')
    print("idx\tlabel\tpredict\tradius\tr_neg\tcorrect\tpred_uc\tcorr_uc\tstage\ttime", file=f, flush=True)

    # iterate through the dataset
    for i in range(args.start, len(dataset)):

        # only certify every args.skip examples, and stop after args.max examples
        if i == args.max:
            break
        if i % args.skip != 0:
            continue

        (x, label) = dataset[i]
        x = x.cuda()
        base_classifier.eval()

        before_time = time()
        radius_neg = -100.
        radius_ = 0.0
        confidences = []
        with torch.cuda.amp.autocast():
            for stage, s in enumerate(smooth[::-1]):
                conf = s.confidence(x, args.N0, args.N, args.batch, skip_p=args.skip_p)
                alpha = args.alpha / (stage + 1)

                pred_certified, radius = conf.certified_radius(alpha)
                if pred_certified >= 0:
                    radius_neg = max([-100.] + [conf.off_class_upper_radius(alpha, pred_certified)
                                                for conf in confidences])
                    radius_ = min(radius, -radius_neg)
                    break
                elif pred_certified == -2:
                    break
                else:
                    confidences.append(conf)

        radius_adj = radius_ / magnitude
        if radius_adj < 0.:
            pred_certified, radius_adj = -1, 0.0

        if pred_certified < 0:
            pred_last = base_classifier(x[None]).argmax(1).item()
        else:
            pred_last = pred_certified

        after_time = time()
        correct = int(pred_certified == label)
        corr_last = int(pred_last == label)

        time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
        print("{}\t{}\t{}\t{:.3}\t{:.3}\t{}\t{}\t{}\t{}\t{}".format(
            i, label, pred_certified, radius_adj, radius_neg, correct, pred_last, corr_last, stage, time_elapsed), file=f, flush=True)

    f.close()
