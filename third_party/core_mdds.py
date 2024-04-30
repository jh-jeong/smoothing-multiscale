# this file is copied from
#   https://github.com/locuslab/smoothing
# originally written by Jeremy Cohen.

import torch
import torch.nn.functional as F

from scipy.stats import norm, binom_test
import numpy as np
from math import ceil
from statsmodels.stats.proportion import proportion_confint, multinomial_proportions_confint


class SmoothedConfidence(object):
    def __init__(self, counts, cAHat: int, sigma: float, n: int, skip_p: float = 0.5):
        self.counts = counts
        self.sigma = sigma
        self.n = n
        self.cAHat = cAHat
        self.skip_p = skip_p

    def confidence_bound(self, alpha: float):
        """ Returns a (1 - alpha) confidence bound on a bernoulli proportion.

        This function uses the Clopper-Pearson method.

        :param NA: the number of "successes"
        :param N: the number of total draws
        :param alpha: the confidence level
        """
        nA = self.counts[self.cAHat].item()
        return proportion_confint(nA, self.n, alpha=2 * alpha, method="beta")

    def certified_radius(self, alpha: float):
        pA_low, pA_high = self.confidence_bound(alpha)
        if pA_low >= self.skip_p:
            return self.cAHat, self.sigma * (norm.ppf(pA_low) - norm.ppf(self.skip_p))
        elif pA_high >= self.skip_p:
            return -2, 0.0
        else:
            return -1, 0.0

    def off_class_upper_radius(self, alpha: float, c: int = None):
        if c is None:
            c = self.cAHat
        mask = np.ones_like(self.counts, dtype=bool)
        mask[c] = False

        rcounts = self._reduce_counts(c)

        if len(rcounts) <= 2:
            pB_high = 1. - self.confidence_bound(alpha)[0]
        else:
            pB_high = multinomial_proportions_confint(rcounts, alpha=2 * alpha)[1:, 1].max()

        return self.sigma * (norm.ppf(pB_high) - norm.ppf(self.skip_p))

    def _reduce_counts(self, c: int):
        counts = self.counts.copy()
        nc = counts[c]
        counts[c] = 0

        r_counts = [nc]
        classes = np.argsort(counts)[::-1]
        reg = 0
        for i, ci in enumerate(classes):
            ni = counts[ci]
            if ni >= 5:
                r_counts.append(ni)
            else:
                reg += ni
                if reg >= 5:
                    r_counts.append(reg)
                    reg = 0
        if reg > 0:
            r_counts[-1] += reg

        return np.array(r_counts, dtype=int)


class Smooth(object):
    """A smoothed classifier g """

    # to abstain, Smooth returns this int
    ABSTAIN = -1

    def __init__(self, base_classifier: torch.nn.Module, num_classes: int, sigma: float):
        """
        :param base_classifier: maps from [batch x channel x height x width] to [batch x num_classes]
        :param num_classes:
        :param sigma: the noise level hyperparameter
        """
        self.base_classifier = base_classifier
        self.num_classes = num_classes
        self.sigma = sigma

    def certify(self, x: torch.tensor, n0: int, n: int, alpha: float, batch_size: int) -> (int, float):
        """ Monte Carlo algorithm for certifying that g's prediction around x is constant within some L2 radius.
        With probability at least 1 - alpha, the class returned by this method will equal g(x), and g's prediction will
        robust within a L2 ball of radius R around x.

        :param x: the input [channel x height x width]
        :param n0: the number of Monte Carlo samples to use for selection
        :param n: the number of Monte Carlo samples to use for estimation
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: (predicted class, certified radius)
                 in the case of abstention, the class will be ABSTAIN and the radius 0.
        """
        self.base_classifier.eval()
        # draw samples of f(x+ epsilon)
        counts_selection = self._sample_noise(x, n0, batch_size)
        # use these samples to take a guess at the top class
        cAHat = counts_selection.argmax().item()
        # draw more samples of f(x + epsilon)
        counts_estimation = self._sample_noise(x, n, batch_size)
        # use these samples to estimate a lower bound on pA
        nA = counts_estimation[cAHat].item()
        pABar = self._lower_confidence_bound(nA, n, alpha)
        if pABar < 0.5:
            return Smooth.ABSTAIN, 0.0
        else:
            radius = self.sigma * norm.ppf(pABar)
            return cAHat, radius

    def confidence(self, x: torch.tensor, n0: int, n: int, batch_size: int, skip_p: float = 0.5):
        """ Monte Carlo algorithm for certifying that g's prediction around x is constant within some L2 radius.
        With probability at least 1 - alpha, the class returned by this method will equal g(x), and g's prediction will
        robust within a L2 ball of radius R around x.

        :param x: the input [channel x height x width]
        :param n0: the number of Monte Carlo samples to use for selection
        :param n: the number of Monte Carlo samples to use for estimation
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: (predicted class, certified radius)
                 in the case of abstention, the class will be ABSTAIN and the radius 0.
        """
        self.base_classifier.eval()
        # draw samples of f(x+ epsilon)
        counts_selection = self._sample_noise(x, n0, batch_size)
        # use these samples to take a guess at the top class
        cAHat = counts_selection.argmax().item()
        # draw more samples of f(x + epsilon)
        counts_estimation = self._sample_noise(x, n, batch_size)
        # use these samples to estimate a lower bound on pA

        conf = SmoothedConfidence(counts_estimation, cAHat, self.sigma, n, skip_p)
        return conf

    def predict(self, x: torch.tensor, n: int, alpha: float, batch_size: int,
                return_top1=False, return_conf=False):
        """ Monte Carlo algorithm for evaluating the prediction of g at x.  With probability at least 1 - alpha, the
        class returned by this method will equal g(x).

        This function uses the hypothesis test described in https://arxiv.org/abs/1610.03944
        for identifying the top category of a multinomial distribution.

        :param x: the input [channel x height x width]
        :param n: the number of Monte Carlo samples to use
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: the predicted class, or ABSTAIN
        """
        self.base_classifier.eval()
        counts = self._sample_noise(x, n, batch_size)
        top2 = counts.argsort()[::-1][:2]
        count1 = counts[top2[0]]
        count2 = counts[top2[1]]
        if binom_test(count1, count1 + count2, p=0.5) > alpha:
            output = Smooth.ABSTAIN
        else:
            output = top2[0]
        if return_top1:
            return output, top2[0]
        if return_conf:
            return output, counts / n
        return output

    def predict_threshold(self, x: torch.tensor, n: int, p: float, batch_size: int):
        """ Monte Carlo algorithm for evaluating the prediction of g at x.  With probability at least 1 - alpha, the
        class returned by this method will equal g(x).

        This function uses the hypothesis test described in https://arxiv.org/abs/1610.03944
        for identifying the top category of a multinomial distribution.

        :param x: the input [channel x height x width]
        :param n: the number of Monte Carlo samples to use
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: the predicted class, or ABSTAIN
        """
        self.base_classifier.eval()
        counts = self._sample_noise(x, n, batch_size)
        cAHat = counts.argmax().item()

        nA = counts[cAHat].item()
        if nA < p * n:
            output = Smooth.ABSTAIN
        else:
            output = cAHat

        conf = SmoothedConfidence(counts, cAHat, self.sigma, n, skip_p=p)
        return output, conf

    def _sample_noise(self, x: torch.tensor, num: int, batch_size) -> np.ndarray:
        """ Sample the base classifier's prediction under noisy corruptions of the input x.

        :param x: the input [channel x width x height]
        :param num: number of samples to collect
        :param batch_size:
        :return: an ndarray[int] of length num_classes containing the per-class counts
        """
        with torch.no_grad():
            counts = np.zeros(self.num_classes, dtype=int)
            for _ in range(ceil(num / batch_size)):
                this_batch_size = min(batch_size, num)
                num -= this_batch_size

                batch = x.repeat((this_batch_size, 1, 1, 1))

                noise = torch.randn_like(batch, device='cuda') * self.sigma
                predictions = self.base_classifier(batch + noise).argmax(1)
                counts += self._count_arr(predictions.cpu().numpy(), self.num_classes)
            return counts

    def _count_arr(self, arr: np.ndarray, length: int) -> np.ndarray:
        counts = np.zeros(length, dtype=int)
        for idx in arr:
            counts[idx] += 1
        return counts

    def _lower_confidence_bound(self, NA: int, N: int, alpha: float) -> float:
        """ Returns a (1 - alpha) lower confidence bound on a bernoulli proportion.

        This function uses the Clopper-Pearson method.

        :param NA: the number of "successes"
        :param N: the number of total draws
        :param alpha: the confidence level
        :return: a lower bound on the binomial proportion which holds true w.p at least (1 - alpha) over the samples
        """
        return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[0]
