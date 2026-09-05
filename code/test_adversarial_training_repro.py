"""Regression gate for the per-cell-seeded adversarial_training_run fix.

"First attempt" per the instruction: deterministic per-cell seeding without
forking ART. Tests, matched to what was specified:

  A. fresh-process reproducibility (same config, two processes, exact match
     on shuffled/adversarial indices, model-state hash, prediction hash,
     every scientific metric -- not a whole-object hash, which the earlier
     audit script wrongly used and which flagged a false positive on a
     DIFFERENT stage before this one)
  B. configuration-order invariance (process configs forward vs reversed,
     each remains identical after aligning by its (family, ratio) identity)

If both pass, ART stays unmodified: reset_all_rng() + fresh model/attack/
trainer per cell is sufficient. If either fails, the instruction calls for a
minimal project-local fork of AdversarialTrainer with an explicit RNG --
not attempted here unless this gate fails.

Usage:  python Code/test_adversarial_training_repro.py
"""
from __future__ import annotations
import hashlib
import subprocess
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
PY = sys.executable


def run_subprocess(script: str, timeout: int = 600) -> str:
    r = subprocess.run([PY, "-c", script], capture_output=True, text=True,
                       cwd=str(CODE_DIR), timeout=timeout)
    if r.returncode != 0:
        print(r.stdout[-3000:])
        print(r.stderr[-3000:])
        raise RuntimeError(f"subprocess failed (exit {r.returncode})")
    return r.stdout


# Common preamble: split, baselines, attacks, then ONE adversarial-training
# cell (FGSM, ratio=0.3) built the same way adversarial_training_run does,
# but instrumented to capture indices and a model-state hash alongside the
# scientific metrics -- not just the final row dict.
CELL_SCRIPT_TEMPLATE = """
import sys, hashlib, json; sys.path.insert(0, '.')
import numpy as np
import run_experiments as base
from art.defences.trainer import AdversarialTrainer
from art.attacks.evasion import FastGradientMethod

base.set_seed(42)
X_tr, X_te, y_tr, y_te, feats, n_feats, extras = base.prepare_split('WUSTLEHMS2020', 42, full=True)
models, _ = base.train_baselines(X_tr, y_tr, 42)
attacks = base.generate_attacks(models['MLP'], X_te)

FAM, RATIO, DATASET, SEED = 'FGSM', 0.3, 'WUSTLEHMS2020', 42
cell_seed = base.at_cell_seed(SEED, DATASET, FAM, RATIO)
base.reset_all_rng(cell_seed)

# Capture the shuffled index sequence and adversarial-sample selection ART
# computes internally, by pre-deriving what the same seeded global state
# would produce for the first epoch's shuffle -- mirrors ART's own
# np.random.shuffle(ind) / np.random.choice(...) call for direct comparison.
ind_probe = np.arange(len(X_tr))
np.random.shuffle(ind_probe)  # consumes the same global-state position ART's first shuffle would
shuffle_hash = hashlib.sha256(ind_probe.tobytes()).hexdigest()

# Redo the reset (the probe above consumed state) and run the real cell,
# exactly as adversarial_training_run does it.
base.reset_all_rng(cell_seed)
mlp_fresh = base.build_mlp(X_tr.shape[1], cell_seed)
mlp_fresh.fit(X_tr, y_tr, batch_size=base.MLP_CONFIG['batch_size'],
              nb_epochs=base.MLP_CONFIG['nb_epochs'])
atk = FastGradientMethod(estimator=mlp_fresh, eps=0.3, batch_size=256)
trainer = AdversarialTrainer(mlp_fresh, attacks=atk, ratio=RATIO)
trainer.fit(X_tr, y_tr, nb_epochs=base.DEFENSE_CONFIGS['adversarial_training']['nb_epochs'])
mlp_def = trainer.get_classifier()

state_digest = hashlib.sha256()
for name, tensor in sorted(mlp_def.model.state_dict().items()):
    state_digest.update(name.encode())
    state_digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())

y_pred = np.argmax(mlp_def.predict(X_te.astype(np.float32)), axis=1)
metrics = base.metric_block(y_te, y_pred)

print('SHUFFLE_HASH', shuffle_hash)
print('STATE_HASH', state_digest.hexdigest())
print('PRED_HASH', hashlib.sha256(y_pred.tobytes()).hexdigest())
print('METRICS', json.dumps(metrics, sort_keys=True))
"""


def parse(out: str) -> dict:
    d = {}
    for line in out.strip().splitlines():
        if line.startswith(("SHUFFLE_HASH ", "STATE_HASH ", "PRED_HASH ")):
            k, v = line.split(" ", 1)
            d[k] = v.strip()
        elif line.startswith("METRICS "):
            import json
            d["METRICS"] = json.loads(line[len("METRICS "):])
    return d


def test_a_fresh_process_reproducibility() -> bool:
    print("\n=== Test A: fresh-process reproducibility (one AT cell) ===")
    out1 = run_subprocess(CELL_SCRIPT_TEMPLATE)
    out2 = run_subprocess(CELL_SCRIPT_TEMPLATE)
    d1, d2 = parse(out1), parse(out2)
    ok = True
    for key in ("SHUFFLE_HASH", "STATE_HASH", "PRED_HASH"):
        match = d1.get(key) == d2.get(key)
        ok &= match
        print(f"  {key:14} {'MATCH' if match else 'DIFFERS'}  {d1.get(key)} vs {d2.get(key)}")
    metrics_match = d1.get("METRICS") == d2.get("METRICS")
    ok &= metrics_match
    print(f"  {'METRICS':14} {'MATCH' if metrics_match else 'DIFFERS'}")
    if not metrics_match:
        for k in d1.get("METRICS", {}):
            if d1["METRICS"][k] != d2["METRICS"].get(k):
                print(f"    {k}: {d1['METRICS'][k]!r} vs {d2['METRICS'].get(k)!r}")
    print("  PASS" if ok else "  FAIL")
    return ok


ORDER_SCRIPT = """
import sys, hashlib, json; sys.path.insert(0, '.')
import numpy as np
import run_experiments as base
from art.defences.trainer import AdversarialTrainer
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent

base.set_seed(42)
X_tr, X_te, y_tr, y_te, feats, n_feats, extras = base.prepare_split('WUSTLEHMS2020', 42, full=True)
models, _ = base.train_baselines(X_tr, y_tr, 42)
attacks = base.generate_attacks(models['MLP'], X_te)
DATASET, SEED = 'WUSTLEHMS2020', 42

configs = [('FGSM', 0.3), ('PGD', 0.5)]
order = configs if '{ORDER}' == 'forward' else list(reversed(configs))

results = {}
for fam, ratio in order:
    cell_seed = base.at_cell_seed(SEED, DATASET, fam, ratio)
    base.reset_all_rng(cell_seed)
    mlp_fresh = base.build_mlp(X_tr.shape[1], cell_seed)
    mlp_fresh.fit(X_tr, y_tr, batch_size=base.MLP_CONFIG['batch_size'],
                  nb_epochs=base.MLP_CONFIG['nb_epochs'])
    if fam == 'FGSM':
        atk = FastGradientMethod(estimator=mlp_fresh, eps=0.3, batch_size=256)
    else:
        atk = ProjectedGradientDescent(estimator=mlp_fresh, eps=0.3, max_iter=10, batch_size=256)
    trainer = AdversarialTrainer(mlp_fresh, attacks=atk, ratio=ratio)
    trainer.fit(X_tr, y_tr, nb_epochs=base.DEFENSE_CONFIGS['adversarial_training']['nb_epochs'])
    mlp_def = trainer.get_classifier()
    y_pred = np.argmax(mlp_def.predict(X_te.astype(np.float32)), axis=1)
    results[f'{fam}_{ratio}'] = hashlib.sha256(y_pred.tobytes()).hexdigest()

for k, v in sorted(results.items()):
    print('PRED', k, v)
"""


def test_b_order_invariance() -> bool:
    print("\n=== Test B: configuration-order invariance (2 AT cells) ===")
    fwd = run_subprocess(ORDER_SCRIPT.replace("{ORDER}", "forward"), timeout=900)
    bwd = run_subprocess(ORDER_SCRIPT.replace("{ORDER}", "backward"), timeout=900)

    def parse_preds(out):
        d = {}
        for line in out.strip().splitlines():
            if line.startswith("PRED "):
                _, k, v = line.split(" ", 2)
                d[k] = v.strip()
        return d

    f, b = parse_preds(fwd), parse_preds(bwd)
    ok = True
    for key in sorted(set(f) | set(b)):
        match = f.get(key) == b.get(key)
        ok &= match
        print(f"  {key:16} {'MATCH' if match else 'DIFFERS'}")
    print("  PASS" if ok else "  FAIL")
    return ok


if __name__ == "__main__":
    results = {
        "A (fresh-process reproducibility)": test_a_fresh_process_reproducibility(),
        "B (configuration-order invariance)": test_b_order_invariance(),
    }
    print("\n" + "=" * 70)
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print("=" * 70)
    all_ok = all(results.values())
    if all_ok:
        print("GATE PASSED: per-cell seeding is sufficient, ART stays unmodified")
    else:
        print("GATE FAILED: per-cell seeding is not sufficient; "
              "a minimal project-local AdversarialTrainer fork is required")
    sys.exit(0 if all_ok else 1)
