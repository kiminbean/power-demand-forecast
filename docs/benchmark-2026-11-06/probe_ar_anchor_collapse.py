"""Reproduce the AR-anchored LSTM collapse and test two remedies on the fixed fold (demand_only).

models/demand_retrain_2026-11-06_raw_ar converged to "AR + constant". This probe retrains the same
model (same data path, split, AR anchor, train_model) for a few epochs and reports train/val loss
against the AR-only val loss, the share of last-hidden-layer units that never activate on validation
inputs, and the spread of the learned correction in MWh:

  default   ReLU head, correction scale s=1 (the pipeline's setting) and s = std of the scaled AR
            residual on train (the network then targets a unit-variance residual)
  leaky     LeakyReLU(0.01) head with s = residual std (a head that cannot die)

Usage (repo root): .venv/bin/python docs/benchmark-2026-11-06/probe_ar_anchor_collapse.py [epochs] [leaky]
Results of the 2026-11-06 runs: probe_ar_anchor_collapse_out.txt
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from data.dataset import TimeSeriesDataset, get_device  # noqa: E402
from models.lstm import LastValueResidual  # noqa: E402
from training.retrain_demand_oot import (  # noqa: E402
    AR_COLUMN, FEATURE_SETS, FIXED_SPLIT, add_ar_anchor, anchor_affine, fill_short_gaps, split,
)
from training.train_production import TRAINING_CONFIG, load_data, train_model  # noqa: E402

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 2
LEAKY = len(sys.argv) > 2 and sys.argv[2] == "leaky"


class Scaled(LastValueResidual):
    def __init__(self, base, s, **kw):
        super().__init__(base, **kw)
        self.s = s

    def forward(self, x):
        anchor = x[:, -1, self.target_idx:self.target_idx + 1] * self.anchor_scale + self.anchor_shift
        return anchor + self.s * self.base(x)


def main():
    device = get_device()
    with tempfile.TemporaryDirectory() as tmp:
        filled = Path(tmp) / "filled.csv"
        fill_short_gaps(ROOT / "data/processed/jeju_hourly_merged.csv", filled)
        df_all, _ = load_data(str(filled), include_weather=False)
    df = df_all[FEATURE_SETS["demand_only"]].copy()
    df, _ = add_ar_anchor(df, FIXED_SPLIT[0])
    tr, va, _ = split(df, *FIXED_SPLIT)
    ar_idx = df.columns.get_loc(AR_COLUMN)
    scale, shift = anchor_affine(tr.values, ar_idx)
    lo, hi = tr.values.min(axis=0), tr.values.max(axis=0)
    span = np.where(hi - lo == 0, 1.0, hi - lo)
    # 타깃 행 t의 잔차 = y_t - (행 t-1의 AR 열 = t 시각 AR 예측), 스케일 단위
    y_s = (tr["power_demand"].values[1:] - lo[0]) / span[0]
    a_s = ((tr[AR_COLUMN].values[:-1] - lo[ar_idx]) / span[ar_idx]) * scale + shift
    s_res = float(np.std(y_s - a_s))
    yv = (va["power_demand"].values[1:] - lo[0]) / span[0]
    av = ((va[AR_COLUMN].values[:-1] - lo[ar_idx]) / span[ar_idx]) * scale + shift
    print(f"residual std (scaled) train {s_res:.5f}  AR-only val MSE {np.mean((yv - av) ** 2):.6f}", flush=True)

    cfg = {**TRAINING_CONFIG, "epochs": EPOCHS}
    variants = [("leaky0.01", s_res)] if LEAKY else [("relu", 1.0), ("relu", s_res)]
    for act, s in variants:
        def wrapper(base, s=s, act=act):
            if act.startswith("leaky"):
                # 실험용: 헤드의 ReLU를 죽지 않는 LeakyReLU로 바꾼다 (공용 모델 코드는 그대로)
                for i, m in enumerate(base.fc):
                    if isinstance(m, torch.nn.ReLU):
                        base.fc[i] = torch.nn.LeakyReLU(float(act[5:]))
            return Scaled(base, s, target_idx=ar_idx, anchor_scale=scale, anchor_shift=shift)
        model, scaler, hist = train_model(tr, va, df.shape[1], device, cfg, seed=42, verbose=0, model_wrapper=wrapper)
        h = hist.history
        acts = []
        hook = model.base.fc[4].register_forward_hook(lambda m, i, o: acts.append(o.detach().cpu()))
        ds = TimeSeriesDataset(scaler.transform(va.values), seq_length=cfg["seq_length"], horizon=1)
        model.eval()
        corr = []
        with torch.no_grad():
            for X, _ in DataLoader(ds, 512):
                X = X.to(device)
                corr.append((model.s * model.base(X)).cpu().numpy().ravel())
        hook.remove()
        a = torch.cat(acts)
        dead = float((a.max(dim=0).values <= 0).float().mean())
        c = np.concatenate(corr) * span[0]
        r_v = (yv[-len(c):] - av[-len(c):]) * span[0]
        rho = np.corrcoef(c, r_v)[0, 1] if c.std() > 0 else float("nan")
        print(f"[{act}] val MAE AR-only {np.mean(np.abs(r_v)):.3f} MWh, model {np.mean(np.abs(r_v - c)):.3f} MWh, "
              f"corr(correction, AR residual) {rho:.3f}", flush=True)
        print(f"s={s:.5f} train_loss {[round(v, 6) for v in h['train_loss']]} val_loss "
              f"{[round(v, 6) for v in h['val_loss']]} dead_units {dead:.2f} corr_std_MWh {c.std():.3f}", flush=True)


if __name__ == "__main__":
    main()
