"""SMP v4.1 서빙용 재학습 (2026-11-05).

기존 v4.1 산출물(models/smp_v4/smp_v41_*)은 그대로 서빙할 수 없다:
- 저장된 가중치는 테스트 MAPE가 가장 좋았던 분할 1, 스케일러·목표 통계는 분할 5의 것이고
  자르기 기준(clip)은 저장되지 않았다. 분할 선택도 테스트 성적으로 해 낙관적이다.
- state_dict().copy()가 얕은 복사라 조기 종료의 최적 가중치가 보존되지 않았다.
- 2022~2024년 데이터만, 24:00이 같은 날 00:00으로 정렬된 상태로 학습했다.

이 스크립트는 같은 v4.1 구조·피처로:
1. 최신 SMP 파일에서 최근 N개월을 walk-forward(1년 학습 → 1개월 테스트)로 평가한다.
   평가는 서빙과 같게 매일 0시 기준 직전 48시간으로 그날 1~24시를 예측하고,
   기준선(전날 같은 시간, 직전값 유지)과 비교한다.
2. 최근 1년으로 최종 모델을 학습해 가중치·정규화 통계·평가 결과를 한 파일에 저장한다.
3. v4.1 평균 MAPE가 전날 같은 시간 기준선보다 낮을 때만 gate_passed=True로 표시한다
   (서빙은 gate_passed인 산출물만 쓴다).

실행: python -m src.smp.models.train_smp_v41_serving --data data/smp/smp_real_epsis.csv
"""

import argparse
import copy
import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.smp.models.smp_v41_features import (
    FEATURE_WARMUP_HOURS, INPUT_HOURS, OUTPUT_HOURS, apply_normalization, build_features, contiguous_starts,
    denormalize, fit_normalization, load_smp_history, midnight_origins,
)
from src.smp.models.train_smp_v4_smp_only import CombinedLossV41, ConfigV41, SMPModelV41

logger = logging.getLogger(__name__)

SEED = 42
DEFAULT_OUTPUT = Path("models/smp_v41_serving/smp_v41_serving.pt")
TRAIN_HOURS = 365 * 24
TEST_DAYS = 30
VAL_FRACTION = 0.1
MAPE_MIN_ABS = 10.0  # 원본 v4.1 평가와 같은 기준 (|실측| <= 10원은 MAPE에서 제외)


class WindowDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray, starts: List[int]):
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.targets = torch.as_tensor(targets, dtype=torch.float32)
        self.starts = starts

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        s = self.starts[i]
        return (self.features[s:s + INPUT_HOURS],
                self.targets[s + INPUT_HOURS:s + INPUT_HOURS + OUTPUT_HOURS])


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def window_starts(datetimes: pd.Series, lo: int, hi: int) -> List[int]:
    """[lo, hi) 안에 입력+목표 창이 들어가고, 피처가 참조하는 앞쪽 336시간까지 끊김이 없는 입력 시작 인덱스.

    피처의 지연값은 '몇 행 앞'으로 계산되므로, 공백을 건너뛰면 실제 시간 간격과 어긋난다.
    """
    length = INPUT_HOURS + OUTPUT_HOURS
    starts = contiguous_starts(datetimes, FEATURE_WARMUP_HOURS + length) + FEATURE_WARMUP_HOURS
    return [int(s) for s in starts if lo <= s and s + length <= hi]


def train_model(norm_features, norm_targets, train_starts, val_starts, config: ConfigV41, device) -> SMPModelV41:
    model = SMPModelV41(input_size=norm_features.shape[1], hidden_size=config.hidden_size,
                        num_layers=config.num_layers, n_heads=config.n_heads,
                        dropout=config.dropout, output_hours=OUTPUT_HOURS).to(device)
    criterion = CombinedLossV41()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=15, factor=0.5)
    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(WindowDataset(norm_features, norm_targets, train_starts),
                              batch_size=config.batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(WindowDataset(norm_features, norm_targets, val_starts),
                            batch_size=config.batch_size, shuffle=False)
    best_loss, best_state, patience = float("inf"), None, 0
    for epoch in range(config.epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            if config.noise_std > 0:
                x = x + torch.randn_like(x) * config.noise_std
            optimizer.zero_grad()
            loss, _ = criterion(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(np.mean([criterion(model(x.to(device)), y.to(device))[0].item() for x, y in val_loader]))
        scheduler.step(val_loss)
        if val_loss < best_loss:
            # 원본은 state_dict().copy()(얕은 복사)라 최적 가중치가 이후 학습에 덮어써졌다.
            best_loss, best_state, patience = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            patience += 1
            if patience >= config.patience:
                break
    model.load_state_dict(best_state)
    logger.info("  trained %d epochs, best val loss %.4f", epoch + 1, best_loss)
    return model


def predict(model, norm_features, starts, stats, device) -> Dict[str, np.ndarray]:
    model.eval()
    xs = torch.stack([torch.as_tensor(norm_features[s:s + INPUT_HOURS], dtype=torch.float32) for s in starts])
    with torch.no_grad():
        out = model(xs.to(device))
    return {k: denormalize(out[k].cpu().numpy(), stats) for k in ("point", "q10", "q90")}


def metrics(pred: np.ndarray, actual: np.ndarray) -> Dict[str, float]:
    mask = np.abs(actual) > MAPE_MIN_ABS
    ss_res = np.sum((actual - pred) ** 2)
    ss_tot = np.sum((actual - actual.mean()) ** 2)
    return {
        "mae": float(np.mean(np.abs(pred - actual))),
        "rmse": float(np.sqrt(np.mean((pred - actual) ** 2))),
        "mape": float(np.mean(np.abs(pred[mask] - actual[mask]) / np.abs(actual[mask])) * 100),
        "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
    }


def split_bounds(datetimes: pd.Series, test_start_time: pd.Timestamp, test_end_time: pd.Timestamp):
    values = datetimes.to_numpy()
    test_lo = int(np.searchsorted(values, np.datetime64(test_start_time - pd.Timedelta(hours=INPUT_HOURS - 1))))
    test_hi = int(np.searchsorted(values, np.datetime64(test_end_time), side="right"))
    train_hi = int(np.searchsorted(values, np.datetime64(test_start_time - pd.Timedelta(hours=INPUT_HOURS - 1))))
    train_lo = max(0, train_hi - TRAIN_HOURS)
    return train_lo, train_hi, test_lo, test_hi


def fit_and_train(df, features, targets, train_lo, train_hi, config, device):
    """[train_lo, train_hi)의 유효 창 중 시간상 앞 90%로 학습, 뒤 10%로 조기 종료한다.

    행이 아니라 '끊김 없는 창' 기준으로 나눈다. 최근 이력에 공백이 많으면 행 기준 뒤 10%에
    유효한 창이 하나도 없을 수 있기 때문이다. 정규화 통계는 학습 창이 덮는 행에서만 계산한다.
    """
    starts = window_starts(df["datetime"], train_lo, train_hi)
    n_val = max(1, int(len(starts) * VAL_FRACTION))
    if len(starts) <= n_val:
        raise ValueError(f"not enough contiguous training windows in rows [{train_lo}, {train_hi}): {len(starts)}")
    train_starts, val_starts = starts[:-n_val], starts[-n_val:]
    covered = np.zeros(len(df), dtype=bool)
    for s in train_starts:
        covered[s:s + INPUT_HOURS + OUTPUT_HOURS] = True
    stats = fit_normalization(features[covered], targets[covered])
    norm_features = apply_normalization(features, stats)
    norm_targets = (targets - stats["target_mean"]) / (stats["target_std"] + 1e-8)
    logger.info("  windows: train %d, val %d (val from %s)", len(train_starts), len(val_starts),
                df["datetime"].iloc[val_starts[0] + INPUT_HOURS - 1])
    model = train_model(norm_features, norm_targets, train_starts, val_starts, config, device)
    return model, stats, norm_features


def run(data_path: Path, output_path: Path, n_splits: int, epochs: int, eval_end: Optional[pd.Timestamp] = None) -> Dict:
    set_seed()
    device = get_device()
    config = ConfigV41()
    config.epochs = epochs
    # 오늘의 하루전 가격(이미 공표)은 학습·평가에 쓰지 않고, 어제 24시 구간 끝(오늘 0시)까지만 쓴다.
    cutoff = pd.Timestamp(datetime.now().date())
    eval_end = min(pd.Timestamp(eval_end), cutoff) if eval_end is not None else cutoff
    df = load_smp_history(data_path, end_at=cutoff)
    features, targets, feature_names = build_features(df)
    logger.info("data %s ~ %s (%d rows), device %s, eval end %s",
                df["datetime"].iloc[0], df["datetime"].iloc[-1], len(df), device, eval_end)

    splits = []
    for k in range(n_splits, 0, -1):
        test_end_time = eval_end - pd.Timedelta(days=TEST_DAYS * (k - 1))
        test_start_time = test_end_time - pd.Timedelta(days=TEST_DAYS)
        train_lo, train_hi, test_lo, test_hi = split_bounds(df["datetime"], test_start_time, test_end_time)
        logger.info("split %d: train %s ~ %s, test %s ~ %s", n_splits - k + 1,
                    df["datetime"].iloc[train_lo], df["datetime"].iloc[train_hi - 1], test_start_time, test_end_time)
        model, stats, norm_features = fit_and_train(df, features, targets, train_lo, train_hi, config, device)
        origins = midnight_origins(df["datetime"], window_starts(df["datetime"], test_lo, test_hi))
        if not origins:
            logger.warning("  no contiguous test days in this split; skipped")
            continue
        actual = np.stack([targets[s + INPUT_HOURS:s + INPUT_HOURS + OUTPUT_HOURS] for s in origins])
        pred = predict(model, norm_features, origins, stats, device)
        seasonal = np.stack([targets[s + INPUT_HOURS - 24:s + INPUT_HOURS] for s in origins])
        persistence = np.stack([np.repeat(targets[s + INPUT_HOURS - 1], OUTPUT_HOURS) for s in origins])
        coverage = float(np.mean((actual >= pred["q10"]) & (actual <= pred["q90"])) * 100)
        split = {
            "test_start": str(test_start_time), "test_end": str(test_end_time), "days": len(origins),
            "v41": {**metrics(pred["point"], actual), "coverage_80": coverage,
                    "interval_width": float(np.mean(pred["q90"] - pred["q10"]))},
            "seasonal_naive_24": metrics(seasonal, actual),
            "persistence": metrics(persistence, actual),
        }
        logger.info("  v4.1 MAPE %.2f%% | seasonal-naive %.2f%% | persistence %.2f%% | coverage %.1f%%",
                    split["v41"]["mape"], split["seasonal_naive_24"]["mape"], split["persistence"]["mape"], coverage)
        splits.append(split)

    if not splits:
        raise ValueError("no evaluable split: not enough contiguous history before eval end")

    def avg(name, key):
        # 분할마다 평가일 수가 다르므로 일수 가중 평균
        return float(np.average([s[name][key] for s in splits], weights=[s["days"] for s in splits]))

    walk_forward = {name: {key: avg(name, key) for key in splits[0][name]}
                    for name in ("v41", "seasonal_naive_24", "persistence")}
    gate_passed = walk_forward["v41"]["mape"] < walk_forward["seasonal_naive_24"]["mape"]
    logger.info("walk-forward avg: v4.1 %.2f%% vs seasonal-naive %.2f%% → gate %s",
                walk_forward["v41"]["mape"], walk_forward["seasonal_naive_24"]["mape"], gate_passed)

    # 최종 모델: 최근 1년(오늘 0시까지)
    n = len(df)
    final_lo = max(0, n - TRAIN_HOURS)
    model, stats, _ = fit_and_train(df, features, targets, final_lo, n, config, device)
    artifact = {
        "version": "4.1-serving",
        "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "model_kwargs": {"input_size": len(feature_names), "hidden_size": config.hidden_size,
                         "num_layers": config.num_layers, "n_heads": config.n_heads,
                         "dropout": config.dropout, "output_hours": OUTPUT_HOURS},
        "feature_names": feature_names,
        "normalization": {k: np.asarray(v).tolist() for k, v in stats.items()},
        "input_hours": INPUT_HOURS,
        "train_range": [str(df["datetime"].iloc[final_lo]), str(df["datetime"].iloc[-1])],
        "data_path": str(data_path),
        "walk_forward": walk_forward,
        "eval_end": str(eval_end),
        "splits": splits,
        "gate_passed": bool(gate_passed),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "seed": SEED,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output_path)
    summary = {k: artifact[k] for k in ("version", "train_range", "walk_forward", "eval_end", "splits", "gate_passed", "trained_at")}
    output_path.with_suffix(".json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("saved %s", output_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=Path("data/smp/jeju_da_smp_hourly.csv"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=ConfigV41.epochs)
    parser.add_argument("--eval-end", type=pd.Timestamp, default=None,
                        help="평가 마지막 시각(기본: 오늘 0시). 최근 이력에 공백이 많으면 완전한 구간 끝으로 지정")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    run(args.data, args.output, args.splits, args.epochs, args.eval_end)


if __name__ == "__main__":
    main()
