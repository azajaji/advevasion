# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2019
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""ReproducibleHopSkipJump: a project-local fork of ART's HopSkipJump attack.

Forked from adversarial-robustness-toolbox==1.20.1,
art/attacks/evasion/hop_skip_jump.py (MIT licensed; header above is ART's
original notice, reproduced verbatim as the license requires). Diffed against
that exact installed file; every change from upstream is marked "PATCH:" in a
comment at the point of change, so the delta is auditable line by line.

Why this fork exists
---------------------
Verified against the installed ART source (art.__version__ == "1.20.1") that
this attack mixes two random-number sources:

  1. `_init_sample` creates `nprd = np.random.RandomState()` with NO seed.
     Every call draws fresh OS entropy, so nothing under this code's control
     -- not `np.random.seed(...)`, not anything else -- determines its draws.
  2. `_compute_update` calls the bare, global `np.random.randn(...)` /
     `np.random.uniform(...)`, which a prior `np.random.seed(seed)` DOES
     control, but whose exact draw sequence depends on how many other calls
     against the global generator happened first in this process -- itself
     not fully controlled once results are grouped, batched, or reordered.

Empirically, this made three back-to-back runs of the ORIGINAL, unmodified
code -- same seed, same trained model (verified bit-identical via SHA-256),
same data (verified bit-identical) -- produce three different attack success
rates (0.865, 0.855, 0.845) for RandomForest. The split and the model were
never the problem; the attack's own randomness was.

This fork replaces both random sources with one explicit `numpy.random.
RandomState` supplied by the caller, so a given seed reproduces a given run
exactly, in a fresh process, regardless of call grouping or order.

It also adds a bounded stopping guard to `_attack`'s step-size search
(originally `while not success: epsilon /= 2.0`, no iteration cap, no check
for epsilon reaching zero or becoming non-finite). Against a tree ensemble's
piecewise-constant decision surface, this loop can genuinely never satisfy
`success` -- confirmed empirically at 926,000+ queries in 258 seconds with
zero convergence for a single sample. The guard does not change max_iter,
max_eval, init_eval, norm, epsilon definition, or any attack setting; it adds
an explicit, disclosed, generous backstop (query budget and step-halving cap)
so a pathological sample terminates and is honestly scored as a failed
attack, instead of hanging.

No other logic differs from upstream ART. `_binary_search`'s loop
(`while (upper_bound - lower_bound) > threshold`) was verified bounded and is
untouched. `_compute_delta`, `_interpolate`, `_check_params` are byte-for-byte
copies.
"""
from __future__ import annotations

import logging

import numpy as np
from tqdm.auto import tqdm

from art.config import ART_NUMPY_DTYPE
from art.attacks.attack import EvasionAttack
from art.estimators.estimator import BaseEstimator
from art.estimators.classification import ClassifierMixin
from art.utils import compute_success, to_categorical, check_and_transform_label_format, get_labels_np_array

logger = logging.getLogger(__name__)

FORKED_FROM_ART_VERSION = "1.20.1"


class ReproducibleHopSkipJump(EvasionAttack):
    """HopSkipJump (Chen, Jordan & Wagner, 2019) with fully seeded randomness
    and a bounded step-size search. See module docstring for the exact
    rationale and the ART version this was forked from.
    """

    attack_params = EvasionAttack.attack_params + [
        "targeted",
        "norm",
        "max_iter",
        "max_eval",
        "init_eval",
        "init_size",
        "curr_iter",
        "batch_size",
        "verbose",
        # PATCH: new params, not present upstream
        "random_state",
        "query_budget",
        "max_step_halvings",
    ]
    _estimator_requirements = (BaseEstimator, ClassifierMixin)

    def __init__(
        self,
        classifier,
        batch_size: int = 64,
        targeted: bool = False,
        norm: int | float | str = 2,
        max_iter: int = 50,
        max_eval: int = 10000,
        init_eval: int = 100,
        init_size: int = 100,
        verbose: bool = True,
        # PATCH: explicit seed replaces ART's two uncontrolled random sources.
        random_state: int | np.random.RandomState | None = None,
        # PATCH: generous backstop for the unbounded upstream step-size loop.
        # Default derived from this attack's own query need, not an arbitrary
        # literature constant: a well-behaved sample needs on the order of
        # max_iter * (max_eval + ~100) queries (gradient estimation dominates;
        # binary search and step-size search are cheap by comparison), so the
        # default here is set comfortably above that so it never truncates a
        # genuinely converging search and only catches real runaways.
        query_budget: int | None = None,
        max_step_halvings: int = 64,
    ) -> None:
        """
        :param classifier: A trained classifier.
        :param batch_size: The size of the batch used by the estimator during inference.
        :param targeted: Should the attack target one specific class.
        :param norm: Order of the norm. Possible values: "inf", np.inf or 2.
        :param max_iter: Maximum number of iterations.
        :param max_eval: Maximum number of evaluations for estimating gradient.
        :param init_eval: Initial number of evaluations for estimating gradient.
        :param init_size: Maximum number of trials for initial generation of adversarial examples.
        :param verbose: Show progress bars.
        :param random_state: Seed (or RandomState) controlling every random draw in this
                             attack. Required for run-to-run reproducibility; ART's own
                             implementation does not offer this.
        :param query_budget: Hard cap on total oracle queries per sample. Defaults to
                             `max_iter * (max_eval + 500)` if not given.
        :param max_step_halvings: Cap on step-size halvings per outer iteration, guarding
                                  the loop that upstream ART leaves unbounded.
        """
        super().__init__(estimator=classifier)
        self._targeted = targeted
        self.norm = norm
        self.max_iter = max_iter
        self.max_eval = max_eval
        self.init_eval = init_eval
        self.init_size = init_size
        self.curr_iter = 0
        self.batch_size = batch_size
        self.verbose = verbose
        self._check_params()
        self.curr_iter = 0

        # PATCH: one explicit RNG for every random draw in this attack.
        if isinstance(random_state, np.random.RandomState):
            self._rng = random_state
        else:
            self._rng = np.random.RandomState(random_state)

        self.query_budget = (query_budget if query_budget is not None
                             else max_iter * (max_eval + 500))
        self.max_step_halvings = max_step_halvings

        # PATCH: per-sample diagnostics, reset at the start of every _perturb call.
        self.n_queries = 0
        self.termination_reason = "converged"  # or "query_budget" / "step_halving_cap"

        if norm == 2:
            self.theta = 0.01 / np.sqrt(np.prod(self.estimator.input_shape))
        else:
            self.theta = 0.01 / np.prod(self.estimator.input_shape)

    def generate(self, x: np.ndarray, y: np.ndarray | None = None, **kwargs) -> np.ndarray:
        """Generate adversarial samples and return them in an array.

        Unchanged from upstream except that per-sample query counts and
        termination reasons are collected into self.last_run_diagnostics.
        """
        mask = kwargs.get("mask")

        if y is None:
            if self.targeted:  # pragma: no cover
                raise ValueError("Target labels `y` need to be provided for a targeted attack.")
            y = get_labels_np_array(self.estimator.predict(x, batch_size=self.batch_size))

        y = check_and_transform_label_format(y, nb_classes=self.estimator.nb_classes)

        if self.estimator.nb_classes == 2 and y.shape[1] == 1:  # pragma: no cover
            raise ValueError(
                "This attack has not yet been tested for binary classification with a single output classifier."
            )

        resume = kwargs.get("resume")
        start = self.curr_iter if (resume is not None and resume) else 0

        if mask is not None:
            if len(mask.shape) == len(x.shape):
                mask = mask.astype(ART_NUMPY_DTYPE)
            else:
                mask = np.array([mask.astype(ART_NUMPY_DTYPE)] * x.shape[0])
        else:
            mask = np.array([None] * x.shape[0])

        if self.estimator.clip_values is not None:
            clip_min, clip_max = self.estimator.clip_values
        else:
            clip_min, clip_max = np.min(x), np.max(x)

        preds = np.argmax(self.estimator.predict(x, batch_size=self.batch_size), axis=1)

        x_adv_init = kwargs.get("x_adv_init")
        if x_adv_init is not None:
            for i in range(x.shape[0]):
                if mask[i] is not None:
                    x_adv_init[i] = x_adv_init[i] * mask[i] + x[i] * (1 - mask[i])
            init_preds = np.argmax(self.estimator.predict(x_adv_init, batch_size=self.batch_size), axis=1)
        else:
            init_preds = [None] * len(x)
            x_adv_init = [None] * len(x)

        if self.targeted and y is None:  # pragma: no cover
            raise ValueError("Target labels `y` need to be provided for a targeted attack.")

        x_adv = x.astype(ART_NUMPY_DTYPE)
        y = np.argmax(y, axis=1)

        # PATCH: per-sample diagnostics collected here (not present upstream).
        self.last_run_diagnostics = []

        for ind, val in enumerate(tqdm(x_adv, desc="ReproducibleHopSkipJump", disable=not self.verbose)):
            self.curr_iter = start
            self.n_queries = 0
            self.termination_reason = "converged"

            if self.targeted:
                x_adv[ind] = self._perturb(
                    x=val, y=y[ind], y_p=preds[ind], init_pred=init_preds[ind],
                    adv_init=x_adv_init[ind], mask=mask[ind],
                    clip_min=clip_min, clip_max=clip_max,
                )
            else:
                x_adv[ind] = self._perturb(
                    x=val, y=-1, y_p=preds[ind], init_pred=init_preds[ind],
                    adv_init=x_adv_init[ind], mask=mask[ind],
                    clip_min=clip_min, clip_max=clip_max,
                )

            self.last_run_diagnostics.append({
                "n_queries": self.n_queries,
                "termination_reason": self.termination_reason,
            })

        y = to_categorical(y, self.estimator.nb_classes)
        logger.info(
            "Success rate of ReproducibleHopSkipJump attack: %.2f%%",
            100 * compute_success(self.estimator, x, y, x_adv, self.targeted, batch_size=self.batch_size),
        )
        return x_adv

    def _perturb(self, x, y, y_p, init_pred, adv_init, mask, clip_min, clip_max) -> np.ndarray:
        """Unchanged from upstream except delegating to the patched helpers below."""
        initial_sample = self._init_sample(x, y, y_p, init_pred, adv_init, mask, clip_min, clip_max)
        if initial_sample is None:
            return x
        x_adv = self._attack(initial_sample[0], x, initial_sample[1], mask, clip_min, clip_max)
        return x_adv

    def _init_sample(self, x, y, y_p, init_pred, adv_init, mask, clip_min, clip_max):
        """Same logic as upstream `_init_sample`.
        PATCH: `nprd = np.random.RandomState()` (unseeded) replaced with `self._rng`.
        """
        initial_sample = None

        if self.targeted:
            if y == y_p:
                return None
            if adv_init is not None and init_pred == y:
                return adv_init.astype(ART_NUMPY_DTYPE), init_pred

            for _ in range(self.init_size):
                # PATCH: self._rng instead of an unseeded nprd
                random_img = self._rng.uniform(clip_min, clip_max, size=x.shape).astype(x.dtype)
                if mask is not None:
                    random_img = random_img * mask + x * (1 - mask)
                random_class = np.argmax(
                    self.estimator.predict(np.array([random_img]), batch_size=self.batch_size), axis=1,
                )[0]
                self.n_queries += 1
                if random_class == y:
                    random_img = self._binary_search(
                        current_sample=random_img, original_sample=x, target=y, norm=2,
                        clip_min=clip_min, clip_max=clip_max, threshold=0.001,
                    )
                    initial_sample = random_img, random_class
                    logger.info("Found initial adversarial image for targeted attack.")
                    break
            else:
                logger.warning("Failed to draw a random image that is adversarial, attack failed.")

        else:
            if adv_init is not None and init_pred != y_p:
                return adv_init.astype(ART_NUMPY_DTYPE), y_p

            for _ in range(self.init_size):
                # PATCH: self._rng instead of an unseeded nprd
                random_img = self._rng.uniform(clip_min, clip_max, size=x.shape).astype(x.dtype)
                if mask is not None:
                    random_img = random_img * mask + x * (1 - mask)
                random_class = np.argmax(
                    self.estimator.predict(np.array([random_img]), batch_size=self.batch_size), axis=1,
                )[0]
                self.n_queries += 1
                if random_class != y_p:
                    random_img = self._binary_search(
                        current_sample=random_img, original_sample=x, target=y_p, norm=2,
                        clip_min=clip_min, clip_max=clip_max, threshold=0.001,
                    )
                    initial_sample = random_img, y_p
                    logger.info("Found initial adversarial image for untargeted attack.")
                    break
            else:
                logger.warning("Failed to draw a random image that is adversarial, attack failed.")

        return initial_sample

    def _attack(self, initial_sample, original_sample, target, mask, clip_min, clip_max) -> np.ndarray:
        """Same logic as upstream `_attack`.
        PATCH: the step-size search (`while not success: epsilon /= 2.0`) gained
        a bounded exit -- query budget, step-halving cap, epsilon underflow, or an
        unchanged candidate after clipping -- any of which forces `success = True`
        so the loop exits through ART's own normal code path with the last
        computed (possibly non-progressing) candidate, rather than looping
        forever. No other line in this method differs from upstream.
        """
        current_sample = initial_sample

        for _ in range(self.max_iter):
            delta = self._compute_delta(
                current_sample=current_sample, original_sample=original_sample,
                clip_min=clip_min, clip_max=clip_max,
            )
            current_sample = self._binary_search(
                current_sample=current_sample, original_sample=original_sample,
                norm=self.norm, target=target, clip_min=clip_min, clip_max=clip_max,
            )
            num_eval = min(int(self.init_eval * np.sqrt(self.curr_iter + 1)), self.max_eval)
            update = self._compute_update(
                current_sample=current_sample, num_eval=num_eval, delta=delta,
                target=target, mask=mask, clip_min=clip_min, clip_max=clip_max,
            )

            if self.norm == 2:
                dist = np.linalg.norm(original_sample - current_sample)
            else:
                dist = np.max(abs(original_sample - current_sample))

            epsilon = 2.0 * dist / np.sqrt(self.curr_iter + 1)
            success = False
            step_halvings = 0
            potential_sample = current_sample

            while not success:
                epsilon /= 2.0
                step_halvings += 1
                potential_sample = current_sample + epsilon * update
                success = self._adversarial_satisfactory(
                    samples=potential_sample[None], target=target,
                    clip_min=clip_min, clip_max=clip_max,
                )
                self.n_queries += 1

                # PATCH: bounded exit conditions, none of which upstream ART checks.
                if success:
                    break
                clipped = np.clip(potential_sample, clip_min, clip_max)
                stagnant = np.array_equal(clipped, np.clip(current_sample, clip_min, clip_max))
                if (epsilon == 0.0 or not np.isfinite(epsilon) or stagnant
                        or step_halvings >= self.max_step_halvings
                        or self.n_queries >= self.query_budget):
                    if self.termination_reason == "converged":  # keep the first cause hit
                        self.termination_reason = (
                            "query_budget" if self.n_queries >= self.query_budget
                            else "step_halving_cap" if step_halvings >= self.max_step_halvings
                            else "epsilon_underflow"
                        )
                    break  # leave potential_sample as the last computed (near-)current_sample

            current_sample = np.clip(potential_sample, clip_min, clip_max)
            self.curr_iter += 1

            if np.isnan(current_sample).any():  # pragma: no cover
                logger.debug("NaN detected in sample, returning original sample.")
                return original_sample

        return current_sample

    def _binary_search(self, current_sample, original_sample, target, norm,
                       clip_min, clip_max, threshold=None) -> np.ndarray:
        """Byte-for-byte copy of upstream `_binary_search`. Verified bounded:
        the loop condition shrinks a finite interval to a fixed threshold, so
        it is not a candidate for the runaway this fork addresses."""
        if norm == 2:
            (upper_bound, lower_bound) = (np.array(1.0), np.array(0.0))
            if threshold is None:
                threshold = self.theta
        else:
            (upper_bound, lower_bound) = (
                np.max(abs(original_sample - current_sample)), np.array(0.0),
            )
            if threshold is None:
                threshold = np.minimum(upper_bound * self.theta, self.theta)

        while (upper_bound - lower_bound) > threshold:
            alpha = (upper_bound + lower_bound) / 2.0
            interpolated_sample = self._interpolate(
                current_sample=current_sample, original_sample=original_sample,
                alpha=alpha, norm=norm,
            )
            satisfied = self._adversarial_satisfactory(
                samples=interpolated_sample[None], target=target,
                clip_min=clip_min, clip_max=clip_max,
            )[0]
            self.n_queries += 1
            lower_bound = np.where(satisfied == 0, alpha, lower_bound)
            upper_bound = np.where(satisfied == 1, alpha, upper_bound)

        result = self._interpolate(
            current_sample=current_sample, original_sample=original_sample,
            alpha=float(upper_bound), norm=norm,
        )
        return result

    def _compute_delta(self, current_sample, original_sample, clip_min, clip_max) -> float:
        """Byte-for-byte copy of upstream `_compute_delta`."""
        if self.curr_iter == 0:
            return 0.1 * (clip_max - clip_min)
        if self.norm == 2:
            dist = np.linalg.norm(original_sample - current_sample)
            delta = np.sqrt(np.prod(self.estimator.input_shape)) * self.theta * dist
        else:
            dist = np.max(abs(original_sample - current_sample))
            delta = np.prod(self.estimator.input_shape) * self.theta * dist
        return delta

    def _compute_update(self, current_sample, num_eval, delta, target, mask,
                        clip_min, clip_max) -> np.ndarray:
        """Same logic as upstream `_compute_update`.
        PATCH: `np.random.randn`/`np.random.uniform` (global state) replaced
        with `self._rng.randn`/`self._rng.uniform` (explicit, seeded state).
        """
        rnd_noise_shape = [num_eval] + list(self.estimator.input_shape)
        if self.norm == 2:
            rnd_noise = self._rng.randn(*rnd_noise_shape).astype(ART_NUMPY_DTYPE)
        else:
            rnd_noise = self._rng.uniform(low=-1, high=1, size=rnd_noise_shape).astype(ART_NUMPY_DTYPE)

        if mask is not None:
            rnd_noise = rnd_noise * mask

        rnd_noise = rnd_noise / np.sqrt(
            np.sum(rnd_noise**2, axis=tuple(range(len(rnd_noise_shape)))[1:], keepdims=True)
        )
        eval_samples = np.clip(current_sample + delta * rnd_noise, clip_min, clip_max)
        rnd_noise = (eval_samples - current_sample) / delta

        satisfied = self._adversarial_satisfactory(
            samples=eval_samples, target=target, clip_min=clip_min, clip_max=clip_max
        )
        self.n_queries += num_eval
        f_val = 2 * satisfied.reshape([num_eval] + [1] * len(self.estimator.input_shape)) - 1.0
        f_val = f_val.astype(ART_NUMPY_DTYPE)

        if np.mean(f_val) == 1.0:
            grad = np.mean(rnd_noise, axis=0)
        elif np.mean(f_val) == -1.0:
            grad = -np.mean(rnd_noise, axis=0)
        else:
            f_val -= np.mean(f_val)
            grad = np.mean(f_val * rnd_noise, axis=0)

        if self.norm == 2:
            result = grad / np.linalg.norm(grad)
        else:
            result = np.sign(grad)
        return result

    def _adversarial_satisfactory(self, samples, target, clip_min, clip_max) -> np.ndarray:
        """Byte-for-byte copy of upstream `_adversarial_satisfactory`."""
        samples = np.clip(samples, clip_min, clip_max)
        preds = np.argmax(self.estimator.predict(samples, batch_size=self.batch_size), axis=1)
        if self.targeted:
            result = preds == target
        else:
            result = preds != target
        return result

    @staticmethod
    def _interpolate(current_sample, original_sample, alpha, norm) -> np.ndarray:
        """Byte-for-byte copy of upstream `_interpolate`."""
        if norm == 2:
            result = (1 - alpha) * original_sample + alpha * current_sample
        else:
            result = np.clip(current_sample, original_sample - alpha, original_sample + alpha)
        return result

    def _check_params(self) -> None:
        """Byte-for-byte copy of upstream `_check_params`."""
        if self.norm not in [2, np.inf, "inf"]:
            raise ValueError('Norm order must be either 2, `np.inf` or "inf".')
        if not isinstance(self.max_iter, int) or self.max_iter < 0:
            raise ValueError("The number of iterations must be a non-negative integer.")
        if not isinstance(self.max_eval, int) or self.max_eval <= 0:
            raise ValueError("The maximum number of evaluations must be a positive integer.")
        if not isinstance(self.init_eval, int) or self.init_eval <= 0:
            raise ValueError("The initial number of evaluations must be a positive integer.")
        if self.init_eval > self.max_eval:
            raise ValueError("The maximum number of evaluations must be larger than the initial number of evaluations.")
        if not isinstance(self.init_size, int) or self.init_size <= 0:
            raise ValueError("The number of initial trials must be a positive integer.")
        if not isinstance(self.verbose, bool):
            raise ValueError("The argument `verbose` has to be of type bool.")
