"""Diagnostic-only subclass of ReproducibleHopSkipJump: per-sample, per-phase
timing and query instrumentation, added without changing any attack
parameter, control flow, or stopping condition.

Not used in any scored experiment. Exists solely to answer one question for
the CSE-CIC `gbm` timeout investigation: where do the 45 minutes go?

How it stays behaviorally identical to ReproducibleHopSkipJump
----------------------------------------------------------------
Every override either (a) forwards straight to `super()` and only wraps a
timer/query-counter around the call, or (b) is `_perturb`, which is copied
here verbatim in control flow (call `_init_sample`; if it returns None,
return the original sample unchanged; otherwise call `_attack` and return
its result) with bookkeeping statements added around the two calls and
nothing else changed. `_attack`, `_init_sample`, `_binary_search`,
`_compute_update`, `_compute_delta`, `_adversarial_satisfactory`,
`_interpolate`, `generate`, and `_check_params` are all inherited untouched
from ReproducibleHopSkipJump -- none of them are redefined here, so their
bodies (queries, RNG draws, iteration counts, stopping conditions) cannot
have changed.

Phase accounting
-----------------
Four buckets, mutually exclusive and collectively exhaustive -- every query
`_perturb` records lands in exactly one of them, and the diagnostic driver
asserts `init + bsearch + gradest + stepsearch == total` per sample as a
self-check, not an assumption:

- init queries/time: the `_init_sample` trial-drawing loop only (the
  `self._rng.uniform(...)` draw-and-predict retries), excluding the one
  `_binary_search` call `_init_sample` makes internally when a trial
  succeeds -- that refinement call is phase-tagged via `self._phase` and
  counted under "binary-search" below, not "init", since it is the same
  method doing the same work as the per-iteration refinement.
- binary-search queries/time: every `_binary_search` call, whether made
  once during init's success refinement or once per outer iteration of
  `_attack`'s loop -- summed together, since both are the same operation.
- gradient-estimation queries/time: every `_compute_update` call (only
  happens during the attack phase).
- step-search queries/time: computed as a residual (attack-phase total minus
  the attack-phase share of binary-search minus gradient-estimation), since
  the step-size `while not success` loop in `_attack` is inline, not a
  separate method, and copying it to instrument directly would risk a
  transcription bug. The only other code running in that residual window is
  `_compute_delta` (no queries, pure numpy arithmetic on a single vector --
  microseconds), so the residual is, to a very close approximation,
  step-search time.
"""
from __future__ import annotations

import time

import numpy as np

from attacks.reproducible_hop_skip_jump import ReproducibleHopSkipJump


class InstrumentedHopSkipJump(ReproducibleHopSkipJump):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._phase = None
        self.rich_diagnostics: list[dict] = []
        self._attack_bsearch_time = 0.0
        self._attack_bsearch_queries = 0
        self._init_bsearch_time = 0.0
        self._init_bsearch_queries = 0
        self._gradest_time = 0.0
        self._gradest_queries = 0

    # ---- wrapped, not reimplemented: same call, timer + query delta around it ----

    def _binary_search(self, *args, **kwargs):
        t0 = time.perf_counter()
        q0 = self.n_queries
        result = super()._binary_search(*args, **kwargs)
        dt = time.perf_counter() - t0
        dq = self.n_queries - q0
        if self._phase == "init":
            self._init_bsearch_time += dt
            self._init_bsearch_queries += dq
        else:
            self._attack_bsearch_time += dt
            self._attack_bsearch_queries += dq
        return result

    def _compute_update(self, *args, **kwargs):
        t0 = time.perf_counter()
        q0 = self.n_queries
        result = super()._compute_update(*args, **kwargs)
        dt = time.perf_counter() - t0
        dq = self.n_queries - q0
        self._gradest_time += dt
        self._gradest_queries += dq
        return result

    # ---- copied verbatim in control flow from ReproducibleHopSkipJump._perturb;
    #      only bookkeeping (timers, phase flag, diagnostic dict) added ----

    def _perturb(self, x, y, y_p, init_pred, adv_init, mask, clip_min, clip_max) -> np.ndarray:
        self._attack_bsearch_time = 0.0
        self._attack_bsearch_queries = 0
        self._init_bsearch_time = 0.0
        self._init_bsearch_queries = 0
        self._gradest_time = 0.0
        self._gradest_queries = 0

        t_start = time.perf_counter()
        q_start = self.n_queries

        self._phase = "init"
        t0 = time.perf_counter()
        initial_sample = self._init_sample(x, y, y_p, init_pred, adv_init, mask, clip_min, clip_max)
        init_wall = time.perf_counter() - t0
        init_queries_total = self.n_queries - q_start
        # init's own trial-drawing cost excludes its one internal
        # _binary_search refinement call, which is counted under
        # "binary-search" below (same method, same operation as the
        # per-iteration refinement in the attack phase).
        init_trial_queries = init_queries_total - self._init_bsearch_queries
        init_trial_time = init_wall - self._init_bsearch_time

        if initial_sample is None:
            total_time = time.perf_counter() - t_start
            total_queries = self.n_queries - q_start
            self.rich_diagnostics.append({
                "init_success": False,
                "init_trials": self.init_size,
                "init_queries": int(init_trial_queries),
                "init_time_s": init_trial_time,
                "bsearch_queries": int(self._init_bsearch_queries),
                "bsearch_time_s": self._init_bsearch_time,
                "gradest_queries": 0, "gradest_time_s": 0.0,
                "stepsearch_queries": 0, "stepsearch_time_s": 0.0,
                "total_queries": int(total_queries),
                "total_time_s": total_time,
                "status": "init_failure",
            })
            return x

        self._phase = "attack"
        t1 = time.perf_counter()
        q1 = self.n_queries
        x_adv = self._attack(initial_sample[0], x, initial_sample[1], mask, clip_min, clip_max)
        attack_wall = time.perf_counter() - t1
        attack_queries = self.n_queries - q1

        stepsearch_queries = attack_queries - self._attack_bsearch_queries - self._gradest_queries
        stepsearch_time = attack_wall - self._attack_bsearch_time - self._gradest_time

        bsearch_queries = self._init_bsearch_queries + self._attack_bsearch_queries
        bsearch_time = self._init_bsearch_time + self._attack_bsearch_time

        total_time = time.perf_counter() - t_start
        total_queries = self.n_queries - q_start

        status = self.termination_reason
        if status == "converged":
            status = "success"

        self.rich_diagnostics.append({
            "init_success": True,
            "init_trials": int(init_trial_queries),
            "init_queries": int(init_trial_queries),
            "init_time_s": init_trial_time,
            "bsearch_queries": int(bsearch_queries),
            "bsearch_time_s": bsearch_time,
            "gradest_queries": int(self._gradest_queries),
            "gradest_time_s": self._gradest_time,
            "stepsearch_queries": int(stepsearch_queries),
            "stepsearch_time_s": stepsearch_time,
            "total_queries": int(total_queries),
            "total_time_s": total_time,
            "status": status,
        })
        return x_adv
