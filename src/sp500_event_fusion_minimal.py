#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单文件、低硬件压力的毕业设计实现：
多源事件文本 + 行情时序融合的标普500方向预测模型。

设计目标：
1. 尽量少文件：训练、评估、预测、示例数据生成全部放在一个脚本里。
2. 尽量低算力：CPU 可跑，不依赖大模型微调，不依赖 GPU。
3. 尽量可答辩：保留文本基线、时序基线、融合模型三组结果，并给出可解释输出。

默认方法：
- 文本：TF-IDF + 少量轻量文本统计特征
- 时序：事件前窗口的收益率/波动率/技术指标统计
- 模型：Logistic Regression（二分类：未来若干 bar 是否上涨）

说明：
- 事件文本默认读取你给出的目录结构，只使用 *_full_summary.txt 作为真实训练文本。
- counterfactual 文本默认不参与训练，原因是它们不是市场真实发生的文本，直接作为监督样本会污染标签。
- 如果你的行情文件是 5 分钟 K 线，建议默认使用：window_bars=24（过去 2 小时），future_bars=12（未来 1 小时）。
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# =========================
# 0. 常量定义
# =========================

EVENT_TYPE_MAP = {
    "1": "Unemployment Insurance Claims",
    "2": "Employment Situation Reports",
    "3": "GDP Advance Releases",
    "4": "FOMC Minutes",
    "5": "Consumer Price Index (CPI) Reports",
    "6": "Producer Price Index (PPI) Reports",
}

DEFAULT_RELEASE_TIME = {
    # 常见美股宏观发布时间的工程化默认值。
    # 这是为了让只有“日期”没有“具体时间”的事件目录也能落地训练。
    "1": "08:30",
    "2": "08:30",
    "3": "08:30",
    "4": "14:00",
    "5": "08:30",
    "6": "08:30",
}

TEXT_META_FEATURE_NAMES = [
    "textmeta__word_count",
    "textmeta__char_count",
    "textmeta__avg_word_len",
    "textmeta__digit_ratio",
    "textmeta__upper_ratio",
    "textmeta__positive_count",
    "textmeta__negative_count",
    "textmeta__uncertainty_count",
    "textmeta__macro_count",
]

TS_FEATURE_NAMES = [
    "ts__ret_mean",
    "ts__ret_std",
    "ts__ret_min",
    "ts__ret_max",
    "ts__ret_sum",
    "ts__ret_skew",
    "ts__ret_kurt",
    "ts__range_mean",
    "ts__range_std",
    "ts__body_mean",
    "ts__body_std",
    "ts__momentum_3",
    "ts__momentum_6",
    "ts__momentum_12",
    "ts__momentum_window",
    "ts__close_vs_ma5",
    "ts__close_vs_ma10",
    "ts__close_vs_ma20",
    "ts__rsi14",
    "ts__atr14",
    "ts__max_drawdown",
    "ts__hour",
    "ts__month",
    "ts__is_intraday_release",
]

POSITIVE_WORDS = {
    "improve", "improved", "improving", "strong", "stronger", "growth", "expanded",
    "expansion", "resilient", "stabilized", "stable", "easing", "cooling", "decline",
    "declined", "deceleration", "moderating", "beat", "beats", "optimistic", "supportive",
}
NEGATIVE_WORDS = {
    "weak", "weaker", "slow", "slowing", "recession", "risk", "risks", "uncertain",
    "uncertainty", "elevated", "inflationary", "tightening", "hawkish", "deterioration",
    "declining", "drop", "fall", "fell", "worse", "volatile", "stress", "loss",
}
UNCERTAINTY_WORDS = {
    "uncertain", "uncertainty", "risk", "risks", "volatile", "volatility", "possible",
    "possibly", "may", "might", "could", "concern", "concerns", "unclear",
}
MACRO_WORDS = {
    "inflation", "employment", "unemployment", "claims", "gdp", "federal", "reserve",
    "rate", "rates", "policy", "committee", "minutes", "consumer", "producer", "price",
}

RANDOM_STATE = 42


# =========================
# 1. 数据结构
# =========================

@dataclass
class TrainConfig:
    events_root: str
    market_file: str
    out_dir: str
    max_features: int = 1000
    window_bars: int = 24
    future_bars: int = 12
    train_ratio: float = 0.7
    valid_ratio: float = 0.15
    c_grid: Tuple[float, ...] = (0.25, 0.5, 1.0, 2.0)
    min_year: Optional[int] = None


# =========================
# 2. 工具函数
# =========================

def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_text(text: str) -> str:
    text = text.replace("\ufeff", " ").replace("\xa0", " ")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def read_text_file(path: Path) -> str:
    for enc in ("utf-8", "utf-8-sig", "gbk", "latin-1"):
        try:
            return normalize_text(path.read_text(encoding=enc))
        except Exception:
            pass
    # 最后的兜底，避免因少数乱码导致整体流程失败
    return normalize_text(path.read_text(errors="ignore"))


def parse_event_date(folder_name: str) -> pd.Timestamp:
    folder_name = folder_name.strip()
    # 支持 20021017 / 2002-10-17 / 2002_10_17 / 2002.10.17
    if re.fullmatch(r"\d{8}", folder_name):
        return pd.to_datetime(folder_name, format="%Y%m%d")
    cleaned = re.sub(r"[._/]", "-", folder_name)
    return pd.to_datetime(cleaned)


def tokenize_english(text: str) -> List[str]:
    return re.findall(r"[A-Za-z][A-Za-z\-']+", text.lower())


def compute_text_meta(texts: Sequence[str]) -> np.ndarray:
    rows = []
    for text in texts:
        tokens = tokenize_english(text)
        word_count = len(tokens)
        char_count = len(text)
        avg_word_len = float(np.mean([len(t) for t in tokens])) if tokens else 0.0
        digit_ratio = sum(ch.isdigit() for ch in text) / max(len(text), 1)
        upper_ratio = sum(ch.isupper() for ch in text) / max(sum(ch.isalpha() for ch in text), 1)
        positive_count = sum(t in POSITIVE_WORDS for t in tokens)
        negative_count = sum(t in NEGATIVE_WORDS for t in tokens)
        uncertainty_count = sum(t in UNCERTAINTY_WORDS for t in tokens)
        macro_count = sum(t in MACRO_WORDS for t in tokens)
        rows.append([
            word_count,
            char_count,
            avg_word_len,
            digit_ratio,
            upper_ratio,
            positive_count,
            negative_count,
            uncertainty_count,
            macro_count,
        ])
    return np.asarray(rows, dtype=np.float32)


def find_first_existing(columns: Iterable[str], candidates: Sequence[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def try_make_onehot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def compute_rsi(close: np.ndarray, period: int = 14) -> float:
    if len(close) < period + 1:
        return 50.0
    diff = np.diff(close)
    gain = np.where(diff > 0, diff, 0.0)
    loss = np.where(diff < 0, -diff, 0.0)
    avg_gain = np.mean(gain[-period:])
    avg_loss = np.mean(loss[-period:])
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return float(100 - (100 / (1 + rs)))


def compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
    if len(close) < 2:
        return 0.0
    prev_close = close[:-1]
    curr_high = high[1:]
    curr_low = low[1:]
    tr = np.maximum(curr_high - curr_low, np.maximum(np.abs(curr_high - prev_close), np.abs(curr_low - prev_close)))
    if len(tr) == 0:
        return 0.0
    return float(np.mean(tr[-period:]))


def compute_max_drawdown(close: np.ndarray) -> float:
    if len(close) == 0:
        return 0.0
    running_max = np.maximum.accumulate(close)
    drawdown = close / np.maximum(running_max, 1e-12) - 1.0
    return float(np.min(drawdown))


def safe_skew(x: np.ndarray) -> float:
    if len(x) < 3:
        return 0.0
    s = pd.Series(x)
    v = s.skew()
    return 0.0 if pd.isna(v) else float(v)


def safe_kurt(x: np.ndarray) -> float:
    if len(x) < 4:
        return 0.0
    s = pd.Series(x)
    v = s.kurt()
    return 0.0 if pd.isna(v) else float(v)


def json_dump(obj: Dict, path: Path) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


# =========================
# 3. 事件文本读取
# =========================

def load_events_from_structure(events_root: str) -> pd.DataFrame:
    root = Path(events_root)
    if not root.exists():
        raise FileNotFoundError(f"事件目录不存在: {root}")

    records = []
    for type_dir in sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name):
        type_id = type_dir.name
        type_name = EVENT_TYPE_MAP.get(type_id, f"EventType_{type_id}")
        default_release = DEFAULT_RELEASE_TIME.get(type_id, "08:30")

        for date_dir in sorted([p for p in type_dir.iterdir() if p.is_dir()], key=lambda p: p.name):
            try:
                event_date = parse_event_date(date_dir.name)
            except Exception:
                continue

            summary_files = list(date_dir.glob("*_full_summary.txt"))
            if not summary_files:
                # 兜底：如果没有标准 summary，就尝试用 chunk summary
                summary_files = list(date_dir.glob("*_chunk_summaries.txt"))
            if not summary_files:
                continue

            text = read_text_file(summary_files[0])
            if not text:
                continue

            publish_time = pd.to_datetime(f"{event_date.date()} {default_release}")
            event_id = f"{type_id}_{event_date.strftime('%Y%m%d')}"

            records.append(
                {
                    "event_id": event_id,
                    "event_type_id": str(type_id),
                    "event_type_name": type_name,
                    "event_date": event_date.normalize(),
                    "publish_time": publish_time,
                    "text": text,
                    "text_path": str(summary_files[0]),
                }
            )

    if not records:
        raise ValueError(
            "没有从事件目录中读取到任何有效文本。\n"
            "请确认路径应当类似: events/processed_events_and_counterfactuals/1/20021017/..._full_summary.txt"
        )

    df = pd.DataFrame(records).sort_values("publish_time").reset_index(drop=True)
    return df


# =========================
# 4. 行情读取
# =========================

def load_market(market_file: str) -> pd.DataFrame:
    path = Path(market_file)
    if not path.exists():
        raise FileNotFoundError(f"行情文件不存在: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    elif suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"暂不支持的行情文件格式: {suffix}")

    ts_col = find_first_existing(df.columns, ["date", "datetime", "timestamp", "time"])
    open_col = find_first_existing(df.columns, ["open", "Open"])
    high_col = find_first_existing(df.columns, ["high", "High"])
    low_col = find_first_existing(df.columns, ["low", "Low"])
    close_col = find_first_existing(df.columns, ["close", "Close"])
    volume_col = find_first_existing(df.columns, ["volume", "Volume", "vol", "Vol"])

    missing = [
        name for name, col in [
            ("timestamp/date", ts_col),
            ("open", open_col),
            ("high", high_col),
            ("low", low_col),
            ("close", close_col),
        ]
        if col is None
    ]
    if missing:
        raise ValueError(f"行情文件缺少必要列: {missing}")

    out = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(df[ts_col]),
            "open": pd.to_numeric(df[open_col], errors="coerce").astype(np.float32),
            "high": pd.to_numeric(df[high_col], errors="coerce").astype(np.float32),
            "low": pd.to_numeric(df[low_col], errors="coerce").astype(np.float32),
            "close": pd.to_numeric(df[close_col], errors="coerce").astype(np.float32),
        }
    )
    if volume_col is not None:
        out["volume"] = pd.to_numeric(df[volume_col], errors="coerce").fillna(0).astype(np.float32)
    else:
        out["volume"] = np.float32(0.0)

    out = out.dropna(subset=["timestamp", "open", "high", "low", "close"]).sort_values("timestamp")
    out = out.drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
    return out


# =========================
# 5. 样本构建
# =========================

def extract_ts_features(window_df: pd.DataFrame, publish_time: pd.Timestamp) -> Dict[str, float]:
    close = window_df["close"].to_numpy(dtype=np.float64)
    open_ = window_df["open"].to_numpy(dtype=np.float64)
    high = window_df["high"].to_numpy(dtype=np.float64)
    low = window_df["low"].to_numpy(dtype=np.float64)

    if len(close) < 2:
        raise ValueError("时序窗口过短，无法提取特征")

    ret = pd.Series(close).pct_change().dropna().to_numpy(dtype=np.float64)
    range_ratio = (high - low) / np.maximum(open_, 1e-12)
    body_ratio = (close - open_) / np.maximum(open_, 1e-12)

    def momentum(n: int) -> float:
        if len(close) <= n:
            return 0.0
        return float(close[-1] / max(close[-1 - n], 1e-12) - 1.0)

    def ma_ratio(n: int) -> float:
        if len(close) < n:
            return 0.0
        return float(close[-1] / max(np.mean(close[-n:]), 1e-12) - 1.0)

    feat = {
        "ts__ret_mean": float(np.mean(ret)) if len(ret) else 0.0,
        "ts__ret_std": float(np.std(ret)) if len(ret) else 0.0,
        "ts__ret_min": float(np.min(ret)) if len(ret) else 0.0,
        "ts__ret_max": float(np.max(ret)) if len(ret) else 0.0,
        "ts__ret_sum": float(np.sum(ret)) if len(ret) else 0.0,
        "ts__ret_skew": safe_skew(ret),
        "ts__ret_kurt": safe_kurt(ret),
        "ts__range_mean": float(np.mean(range_ratio)),
        "ts__range_std": float(np.std(range_ratio)),
        "ts__body_mean": float(np.mean(body_ratio)),
        "ts__body_std": float(np.std(body_ratio)),
        "ts__momentum_3": momentum(3),
        "ts__momentum_6": momentum(6),
        "ts__momentum_12": momentum(12),
        "ts__momentum_window": float(close[-1] / max(close[0], 1e-12) - 1.0),
        "ts__close_vs_ma5": ma_ratio(5),
        "ts__close_vs_ma10": ma_ratio(10),
        "ts__close_vs_ma20": ma_ratio(20),
        "ts__rsi14": compute_rsi(close, 14) / 100.0,
        "ts__atr14": compute_atr(high, low, close, 14) / max(close[-1], 1e-12),
        "ts__max_drawdown": compute_max_drawdown(close),
        "ts__hour": float(publish_time.hour + publish_time.minute / 60.0),
        "ts__month": float(publish_time.month),
        "ts__is_intraday_release": float(int(9 <= publish_time.hour < 16)),
    }
    return feat


def build_aligned_samples(
    events_df: pd.DataFrame,
    market_df: pd.DataFrame,
    window_bars: int = 24,
    future_bars: int = 12,
    min_year: Optional[int] = None,
) -> pd.DataFrame:
    market_df = market_df.copy().sort_values("timestamp").reset_index(drop=True)
    ts_values = market_df["timestamp"].to_numpy(dtype="datetime64[ns]")

    if min_year is not None:
        events_df = events_df[events_df["publish_time"].dt.year >= min_year].copy()

    min_ts = market_df["timestamp"].min()
    max_ts = market_df["timestamp"].max()
    events_df = events_df[(events_df["publish_time"] >= min_ts) & (events_df["publish_time"] <= max_ts)].copy()

    rows = []
    for _, ev in events_df.iterrows():
        publish_time = pd.Timestamp(ev["publish_time"])
        idx = int(np.searchsorted(ts_values, np.datetime64(publish_time), side="left"))

        # feature window 必须完全位于事件前，label 需要足够未来 bar
        if idx < window_bars:
            continue
        if idx + future_bars - 1 >= len(market_df):
            continue

        anchor_bar = market_df.iloc[idx]
        end_bar = market_df.iloc[idx + future_bars - 1]

        # 为了让“未来未来1小时/12个bar”更纯净，不跨日
        if anchor_bar["timestamp"].date() != end_bar["timestamp"].date():
            continue

        window_df = market_df.iloc[idx - window_bars : idx].copy()
        if len(window_df) != window_bars:
            continue

        ts_feat = extract_ts_features(window_df, publish_time)
        future_return = float(end_bar["close"] / max(anchor_bar["open"], 1e-12) - 1.0)
        label_up = int(future_return > 0)

        row = {
            "event_id": ev["event_id"],
            "event_type_id": ev["event_type_id"],
            "event_type_name": ev["event_type_name"],
            "event_date": pd.Timestamp(ev["event_date"]),
            "publish_time": publish_time,
            "anchor_time": pd.Timestamp(anchor_bar["timestamp"]),
            "future_end_time": pd.Timestamp(end_bar["timestamp"]),
            "future_return": future_return,
            "label_up": label_up,
            "text": ev["text"],
            "text_path": ev["text_path"],
        }
        row.update(ts_feat)
        rows.append(row)

    if not rows:
        raise ValueError(
            "对齐后没有得到任何可训练样本。\n"
            "可能原因：\n"
            "1) 事件时间与行情时间不重叠；\n"
            "2) window_bars/future_bars 太大；\n"
            "3) 事件目录日期和行情时间格式不一致。"
        )

    out = pd.DataFrame(rows).sort_values("anchor_time").reset_index(drop=True)
    return out


# =========================
# 6. 特征变换与训练
# =========================

def time_split_indices(n: int, train_ratio: float, valid_ratio: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n < 30:
        raise ValueError(f"样本太少（{n} 条），建议至少保证 30 条以上再训练")
    train_end = max(1, int(n * train_ratio))
    valid_end = max(train_end + 1, int(n * (train_ratio + valid_ratio)))
    valid_end = min(valid_end, n - 1)

    train_idx = np.arange(0, train_end)
    valid_idx = np.arange(train_end, valid_end)
    test_idx = np.arange(valid_end, n)
    if len(valid_idx) == 0 or len(test_idx) == 0:
        raise ValueError("时间切分失败：验证集或测试集为空，请调整 train_ratio / valid_ratio")
    return train_idx, valid_idx, test_idx


def prepare_feature_artifacts(train_df: pd.DataFrame, max_features: int = 1000):
    min_df = 2 if len(train_df) >= 20 else 1
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=max_features,
        min_df=min_df,
        sublinear_tf=True,
    )
    vectorizer.fit(train_df["text"].astype(str).tolist())

    text_meta_scaler = StandardScaler()
    ts_scaler = StandardScaler()
    event_encoder = try_make_onehot_encoder()

    train_text_meta = compute_text_meta(train_df["text"].astype(str).tolist())
    text_meta_scaler.fit(train_text_meta)

    train_ts = train_df[TS_FEATURE_NAMES].to_numpy(dtype=np.float32)
    ts_scaler.fit(train_ts)

    event_encoder.fit(train_df[["event_type_name"]])

    return {
        "vectorizer": vectorizer,
        "text_meta_scaler": text_meta_scaler,
        "ts_scaler": ts_scaler,
        "event_encoder": event_encoder,
    }


def transform_feature_views(df: pd.DataFrame, feature_artifacts: Dict):
    vectorizer = feature_artifacts["vectorizer"]
    text_meta_scaler = feature_artifacts["text_meta_scaler"]
    ts_scaler = feature_artifacts["ts_scaler"]
    event_encoder = feature_artifacts["event_encoder"]

    texts = df["text"].astype(str).tolist()
    x_tfidf = vectorizer.transform(texts)

    x_text_meta = text_meta_scaler.transform(compute_text_meta(texts)).astype(np.float32)
    x_ts = ts_scaler.transform(df[TS_FEATURE_NAMES].to_numpy(dtype=np.float32)).astype(np.float32)
    x_event = event_encoder.transform(df[["event_type_name"]])

    x_text_meta_sparse = sparse.csr_matrix(x_text_meta)
    x_ts_sparse = sparse.csr_matrix(x_ts)

    x_text = sparse.hstack([x_tfidf, x_text_meta_sparse, x_event], format="csr")
    x_ts_only = sparse.hstack([x_ts_sparse, x_event], format="csr")
    x_fusion = sparse.hstack([x_tfidf, x_text_meta_sparse, x_ts_sparse, x_event], format="csr")

    tfidf_names = [f"tfidf__{x}" for x in vectorizer.get_feature_names_out().tolist()]
    try:
        event_names = event_encoder.get_feature_names_out(["event_type_name"]).tolist()
    except Exception:
        categories = event_encoder.categories_[0].tolist()
        event_names = [f"event_type_name_{x}" for x in categories]

    feature_names = {
        "text": tfidf_names + TEXT_META_FEATURE_NAMES + event_names,
        "ts": TS_FEATURE_NAMES + event_names,
        "fusion": tfidf_names + TEXT_META_FEATURE_NAMES + TS_FEATURE_NAMES + event_names,
    }

    return {
        "text": x_text,
        "ts": x_ts_only,
        "fusion": x_fusion,
        "feature_names": feature_names,
    }


def fit_best_logistic(
    x_train,
    y_train: np.ndarray,
    x_valid,
    y_valid: np.ndarray,
    c_grid: Sequence[float],
) -> Dict:
    best = None
    for c in c_grid:
        model = LogisticRegression(
            C=float(c),
            solver="liblinear",
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
        model.fit(x_train, y_train)
        valid_proba = model.predict_proba(x_valid)[:, 1]
        metrics = compute_metrics(y_valid, valid_proba)
        metrics["C"] = float(c)
        if best is None or metrics["auc"] > best["metrics"]["auc"]:
            best = {"model": model, "metrics": metrics}
    return best


def compute_metrics(y_true: np.ndarray, proba: np.ndarray) -> Dict[str, float]:
    pred = (proba >= 0.5).astype(int)
    # 兼容极端小样本场景
    if len(np.unique(y_true)) == 1:
        auc = 0.5
    else:
        auc = float(roc_auc_score(y_true, proba))
    acc = float(accuracy_score(y_true, pred))
    f1 = float(f1_score(y_true, pred, zero_division=0))
    precision = float(precision_score(y_true, pred, zero_division=0))
    recall = float(recall_score(y_true, pred, zero_division=0))
    cm = confusion_matrix(y_true, pred)
    tn, fp, fn, tp = (cm.ravel().tolist() if cm.size == 4 else [0, 0, 0, 0])
    return {
        "auc": auc,
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "positive_rate_pred": float(pred.mean()),
    }


def add_return_metrics(metrics: Dict, returns: np.ndarray, proba: np.ndarray) -> Dict:
    out = dict(metrics)
    pred_up = proba >= 0.5
    out["avg_future_return_all"] = float(np.mean(returns)) if len(returns) else 0.0
    out["avg_future_return_pred_up"] = float(np.mean(returns[pred_up])) if pred_up.any() else 0.0
    out["avg_future_return_pred_down"] = float(np.mean(returns[~pred_up])) if (~pred_up).any() else 0.0
    if len(returns) > 1:
        out["prob_return_corr"] = float(np.corrcoef(proba, returns)[0, 1]) if np.std(proba) > 0 and np.std(returns) > 0 else 0.0
    else:
        out["prob_return_corr"] = 0.0
    return out


def build_global_importance(feature_names: List[str], model: LogisticRegression, topn: int = 50) -> pd.DataFrame:
    coef = model.coef_.ravel()
    df = pd.DataFrame({"feature": feature_names, "coef": coef, "abs_coef": np.abs(coef)})
    return df.sort_values("abs_coef", ascending=False).head(topn).reset_index(drop=True)


def local_explanation(row_x, feature_names: List[str], model: LogisticRegression, topn: int = 10) -> Dict[str, List[Tuple[str, float]]]:
    coef = model.coef_.ravel()
    values = row_x.toarray().ravel() if sparse.issparse(row_x) else np.asarray(row_x).ravel()
    contrib = values * coef
    nonzero = np.where(np.abs(contrib) > 1e-12)[0]
    items = [(feature_names[i], float(contrib[i])) for i in nonzero]
    items_sorted = sorted(items, key=lambda x: abs(x[1]), reverse=True)

    top_positive = [(n, v) for n, v in items_sorted if v > 0][:topn]
    top_negative = [(n, v) for n, v in items_sorted if v < 0][:topn]
    top_words = [(n.replace("tfidf__", ""), v) for n, v in items_sorted if n.startswith("tfidf__")][:topn]
    key_indicators = [
        (n.replace("ts__", ""), v)
        for n, v in items_sorted
        if n.startswith("ts__") or n.startswith("event_type_name_")
    ][:topn]
    return {
        "top_positive": top_positive,
        "top_negative": top_negative,
        "top_words": top_words,
        "key_indicators": key_indicators,
    }


def ensure_binary_trainable(y: np.ndarray, name: str) -> None:
    uniq = np.unique(y)
    if len(uniq) < 2:
        raise ValueError(f"{name} 只有一个类别 {uniq.tolist()}，无法训练二分类模型。\n请尝试：\n1) 扩大时间范围；\n2) 调整 future_bars；\n3) 确认事件与行情对齐无误。")


# =========================
# 7. 训练主流程
# =========================

def train_pipeline(cfg: TrainConfig) -> Dict:
    out_dir = Path(cfg.out_dir)
    safe_mkdir(out_dir)

    print("[1/4] 读取事件文本...")
    events_df = load_events_from_structure(cfg.events_root)
    print(f"  事件总数: {len(events_df)}")

    print("[2/4] 读取行情数据...")
    market_df = load_market(cfg.market_file)
    print(f"  行情总行数: {len(market_df)}")

    if cfg.min_year is None:
        cfg.min_year = int(market_df["timestamp"].dt.year.min())

    print("[3/4] 对齐并生成样本...")
    samples_df = build_aligned_samples(
        events_df=events_df,
        market_df=market_df,
        window_bars=cfg.window_bars,
        future_bars=cfg.future_bars,
        min_year=cfg.min_year,
    )
    print(f"  对齐后样本数: {len(samples_df)}")

    samples_df.to_csv(out_dir / "aligned_samples.csv", index=False, encoding="utf-8-sig")

    train_idx, valid_idx, test_idx = time_split_indices(len(samples_df), cfg.train_ratio, cfg.valid_ratio)
    train_df = samples_df.iloc[train_idx].reset_index(drop=True)
    valid_df = samples_df.iloc[valid_idx].reset_index(drop=True)
    test_df = samples_df.iloc[test_idx].reset_index(drop=True)

    ensure_binary_trainable(train_df["label_up"].to_numpy(), "训练集")
    ensure_binary_trainable(valid_df["label_up"].to_numpy(), "验证集")
    ensure_binary_trainable(test_df["label_up"].to_numpy(), "测试集")

    print("[4/4] 训练 Text-only / TS-only / Fusion 三组模型...")
    feature_artifacts = prepare_feature_artifacts(train_df, cfg.max_features)

    train_views = transform_feature_views(train_df, feature_artifacts)
    valid_views = transform_feature_views(valid_df, feature_artifacts)
    test_views = transform_feature_views(test_df, feature_artifacts)

    y_train = train_df["label_up"].to_numpy(dtype=int)
    y_valid = valid_df["label_up"].to_numpy(dtype=int)
    y_test = test_df["label_up"].to_numpy(dtype=int)

    results = {}
    models = {}
    for view_name in ["text", "ts", "fusion"]:
        best = fit_best_logistic(
            train_views[view_name],
            y_train,
            valid_views[view_name],
            y_valid,
            cfg.c_grid,
        )
        model = best["model"]
        models[view_name] = model

        valid_proba = model.predict_proba(valid_views[view_name])[:, 1]
        test_proba = model.predict_proba(test_views[view_name])[:, 1]

        results[view_name] = {
            "valid": add_return_metrics(
                compute_metrics(y_valid, valid_proba),
                valid_df["future_return"].to_numpy(dtype=float),
                valid_proba,
            ),
            "test": add_return_metrics(
                compute_metrics(y_test, test_proba),
                test_df["future_return"].to_numpy(dtype=float),
                test_proba,
            ),
            "best_C": best["metrics"]["C"],
        }

    # 选择验证集 AUC 最优模型作为默认推理模型
    best_model_name = max(results.keys(), key=lambda k: results[k]["valid"]["auc"])
    best_model = models[best_model_name]
    best_feature_names = test_views["feature_names"][best_model_name]
    importance_df = build_global_importance(best_feature_names, best_model, topn=100)
    importance_df.to_csv(out_dir / "global_importance.csv", index=False, encoding="utf-8-sig")

    bundle = {
        "config": asdict(cfg),
        "feature_artifacts": feature_artifacts,
        "models": models,
        "best_model_name": best_model_name,
        "feature_names": test_views["feature_names"],
        "ts_feature_names": TS_FEATURE_NAMES,
        "text_meta_feature_names": TEXT_META_FEATURE_NAMES,
        "event_type_map": EVENT_TYPE_MAP,
        "default_release_time": DEFAULT_RELEASE_TIME,
        "train_date_range": [str(train_df["anchor_time"].min()), str(train_df["anchor_time"].max())],
        "valid_date_range": [str(valid_df["anchor_time"].min()), str(valid_df["anchor_time"].max())],
        "test_date_range": [str(test_df["anchor_time"].min()), str(test_df["anchor_time"].max())],
    }
    joblib.dump(bundle, out_dir / "model_bundle.joblib")

    summary = {
        "sample_count": int(len(samples_df)),
        "train_count": int(len(train_df)),
        "valid_count": int(len(valid_df)),
        "test_count": int(len(test_df)),
        "best_model_name": best_model_name,
        "results": results,
    }
    json_dump(summary, out_dir / "metrics.json")

    print("训练完成。")
    print(f"默认推理模型: {best_model_name}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


# =========================
# 8. 单条预测
# =========================

def infer_event_type_name(event_type: str) -> str:
    event_type = str(event_type).strip()
    if event_type in EVENT_TYPE_MAP:
        return EVENT_TYPE_MAP[event_type]
    # 允许用户直接传英文名称
    return event_type


def normalize_publish_time(raw_publish_time: str, event_type_name: str) -> pd.Timestamp:
    ts = pd.to_datetime(raw_publish_time)
    # 如果用户只给日期，则按默认发布时间补上
    if ts.hour == 0 and ts.minute == 0 and len(str(raw_publish_time).strip()) <= 10:
        type_id = None
        for k, v in EVENT_TYPE_MAP.items():
            if v == event_type_name:
                type_id = k
                break
        release = DEFAULT_RELEASE_TIME.get(type_id or "", "08:30")
        ts = pd.to_datetime(f"{ts.date()} {release}")
    return ts


def build_single_sample(text: str, publish_time: pd.Timestamp, event_type_name: str, market_df: pd.DataFrame, window_bars: int, future_bars: int = 12) -> pd.DataFrame:
    # 训练时 future_bars 用于检查标签；预测时未来标签未知，因此这里只构造过去窗口特征。
    ts_values = market_df["timestamp"].to_numpy(dtype="datetime64[ns]")
    idx = int(np.searchsorted(ts_values, np.datetime64(publish_time), side="left"))
    if idx < window_bars:
        raise ValueError("预测失败：该事件前的可用行情窗口不足，请减小 window_bars 或检查发布时间")
    if idx >= len(market_df):
        raise ValueError("预测失败：发布时间晚于行情文件末尾")

    anchor_bar = market_df.iloc[idx]
    if idx + future_bars - 1 < len(market_df):
        end_bar = market_df.iloc[idx + future_bars - 1]
        future_return = float(end_bar["close"] / max(anchor_bar["open"], 1e-12) - 1.0)
        label_up = int(future_return > 0)
        future_end_time = pd.Timestamp(end_bar["timestamp"])
    else:
        future_return = np.nan
        label_up = np.nan
        future_end_time = pd.NaT

    window_df = market_df.iloc[idx - window_bars : idx].copy()
    ts_feat = extract_ts_features(window_df, publish_time)

    row = {
        "event_id": "predict_one",
        "event_type_id": "unknown",
        "event_type_name": event_type_name,
        "event_date": publish_time.normalize(),
        "publish_time": publish_time,
        "anchor_time": pd.Timestamp(anchor_bar["timestamp"]),
        "future_end_time": future_end_time,
        "future_return": future_return,
        "label_up": label_up,
        "text": normalize_text(text),
        "text_path": "<manual_input>",
    }
    row.update(ts_feat)
    return pd.DataFrame([row])


def predict_one(
    bundle_path: str,
    market_file: str,
    event_text: str,
    publish_time: str,
    event_type: str,
    model_name: Optional[str] = None,
    out_json: Optional[str] = None,
) -> Dict:
    bundle = joblib.load(bundle_path)
    market_df = load_market(market_file)

    event_type_name = infer_event_type_name(event_type)
    publish_ts = normalize_publish_time(publish_time, event_type_name)
    window_bars = int(bundle["config"]["window_bars"])
    future_bars = int(bundle["config"].get("future_bars", 12))

    df_one = build_single_sample(
        text=event_text,
        publish_time=publish_ts,
        event_type_name=event_type_name,
        market_df=market_df,
        window_bars=window_bars,
        future_bars=future_bars,
    )

    views = transform_feature_views(df_one, bundle["feature_artifacts"])
    chosen_model_name = model_name or bundle["best_model_name"]
    model = bundle["models"][chosen_model_name]
    x = views[chosen_model_name]
    proba_up = float(model.predict_proba(x)[0, 1])
    pred_label = int(proba_up >= 0.5)

    explanation = local_explanation(
        x[0],
        bundle["feature_names"][chosen_model_name],
        model,
        topn=10,
    )

    result = {
        "model_name": chosen_model_name,
        "pred_label": pred_label,
        "pred_direction": "UP" if pred_label == 1 else "DOWN",
        "pred_score": proba_up,
        "confidence": float(abs(proba_up - 0.5) * 2),
        "anchor_time": str(df_one.loc[0, "anchor_time"]),
        "future_end_time_if_available": str(df_one.loc[0, "future_end_time"]),
        "top_words": explanation["top_words"],
        "key_indicators": explanation["key_indicators"],
        "top_positive_features": explanation["top_positive"],
        "top_negative_features": explanation["top_negative"],
    }

    if out_json:
        json_dump(result, Path(out_json))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


# =========================
# 9. 示例数据生成（方便你先跑通）
# =========================

def make_demo_data(out_dir: str) -> None:
    """
    生成一个极小的可运行 demo，方便先验证脚本全链路没有问题。
    不代表真实金融规律，只用于跑通工程。
    """
    root = Path(out_dir)
    event_root = root / "events" / "processed_events_and_counterfactuals"
    market_file = root / "sp500_demo_5min.csv"
    safe_mkdir(event_root)

    # 1) 造一份简化 5 分钟 K 线（只生成少量日期）
    rng = np.random.default_rng(RANDOM_STATE)
    days = pd.bdate_range("2023-01-03", "2024-12-31", freq="B")
    timestamps = []
    opens, highs, lows, closes = [], [], [], []
    price = 3800.0

    for day in days:
        intraday = pd.date_range(f"{day.date()} 09:30", f"{day.date()} 16:00", freq="5min")
        intraday = intraday[:-1]  # 保持 78 个 5min bar 左右
        day_shock = rng.normal(0, 0.002)
        for ts in intraday:
            open_p = price
            ret = rng.normal(day_shock / max(len(intraday), 1), 0.0009)
            close_p = max(1000.0, open_p * (1 + ret))
            high_p = max(open_p, close_p) * (1 + abs(rng.normal(0, 0.0005)))
            low_p = min(open_p, close_p) * (1 - abs(rng.normal(0, 0.0005)))
            timestamps.append(ts)
            opens.append(open_p)
            highs.append(high_p)
            lows.append(low_p)
            closes.append(close_p)
            price = close_p

    market_df = pd.DataFrame(
        {
            "date": timestamps,
            "Open": np.round(opens, 2),
            "High": np.round(highs, 2),
            "Low": np.round(lows, 2),
            "Close": np.round(closes, 2),
        }
    )
    market_df.to_csv(market_file, index=False, encoding="utf-8-sig")

    # 2) 造一批事件 summary
    text_templates = {
        "5": [
            "Inflation cooled more than expected and consumer prices moderated broadly.",
            "Consumer prices remained sticky and inflation pressures stayed elevated.",
        ],
        "4": [
            "The minutes signaled a cautious but supportive tone with easing concerns.",
            "The minutes were more hawkish with elevated inflation risks and tighter policy outlook.",
        ],
        "2": [
            "Employment growth was resilient and labor market conditions improved.",
            "Labor market momentum weakened and uncertainty about growth increased.",
        ],
    }

    demo_dates = pd.bdate_range("2023-02-01", "2024-12-15", freq="10B")
    type_cycle = ["5", "4", "2"]

    for i, day in enumerate(demo_dates):
        type_id = type_cycle[i % len(type_cycle)]
        date_dir = event_root / type_id / day.strftime("%Y%m%d")
        safe_mkdir(date_dir)
        good = (i % 2 == 0)
        text = text_templates[type_id][0 if good else 1]
        # 顺手加一点随机词，避免 tf-idf 完全重复
        noise = " growth resilient supportive " if good else " risk uncertainty weak "
        content = f"{text} {noise} report index {i}."
        (date_dir / f"{day.strftime('%Y%m%d')}.txt_full_summary.txt").write_text(content, encoding="utf-8")

    print(f"示例事件目录已生成: {event_root}")
    print(f"示例行情文件已生成: {market_file}")
    print("下一步可直接运行：")
    print(
        f"python {Path(__file__).name} train --events-root \"{event_root}\" --market-file \"{market_file}\" --out-dir \"{root / 'artifacts'}\""
    )


# =========================
# 10. CLI
# =========================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="低硬件压力的多源事件文本 + 时序融合标普500预测模型（单文件版）"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_demo = sub.add_parser("demo-data", help="生成一份可跑通流程的演示数据")
    p_demo.add_argument("--out-dir", required=True, help="演示数据输出目录")

    p_build = sub.add_parser("build-samples", help="只做事件-行情对齐，输出 aligned_samples.csv")
    p_build.add_argument("--events-root", required=True, help="事件根目录，例如 processed_events_and_counterfactuals")
    p_build.add_argument("--market-file", required=True, help="行情文件路径，支持 csv/xlsx/parquet")
    p_build.add_argument("--out-csv", required=True, help="样本输出 csv")
    p_build.add_argument("--window-bars", type=int, default=24)
    p_build.add_argument("--future-bars", type=int, default=12)
    p_build.add_argument("--min-year", type=int, default=None)

    p_train = sub.add_parser("train", help="训练 Text-only / TS-only / Fusion 三组模型")
    p_train.add_argument("--events-root", required=True)
    p_train.add_argument("--market-file", required=True)
    p_train.add_argument("--out-dir", required=True)
    p_train.add_argument("--max-features", type=int, default=1000)
    p_train.add_argument("--window-bars", type=int, default=24)
    p_train.add_argument("--future-bars", type=int, default=12)
    p_train.add_argument("--train-ratio", type=float, default=0.7)
    p_train.add_argument("--valid-ratio", type=float, default=0.15)
    p_train.add_argument("--min-year", type=int, default=None)

    p_predict = sub.add_parser("predict", help="使用已训练 bundle 对单条事件做预测")
    p_predict.add_argument("--bundle", required=True, help="model_bundle.joblib 路径")
    p_predict.add_argument("--market-file", required=True, help="行情文件路径")
    p_predict.add_argument("--event-text", required=True, help="事件文本")
    p_predict.add_argument("--publish-time", required=True, help="发布时间，如 2024-03-20 08:30:00")
    p_predict.add_argument("--event-type", required=True, help="事件类型 id(1-6) 或英文名称")
    p_predict.add_argument("--model-name", default=None, choices=[None, "text", "ts", "fusion"], help="指定使用哪个模型视图")
    p_predict.add_argument("--out-json", default=None, help="可选：保存预测结果 json")

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "demo-data":
        make_demo_data(args.out_dir)
        return 0

    if args.command == "build-samples":
        events_df = load_events_from_structure(args.events_root)
        market_df = load_market(args.market_file)
        samples_df = build_aligned_samples(
            events_df=events_df,
            market_df=market_df,
            window_bars=args.window_bars,
            future_bars=args.future_bars,
            min_year=args.min_year,
        )
        out_csv = Path(args.out_csv)
        safe_mkdir(out_csv.parent)
        samples_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"已输出对齐样本: {out_csv}")
        print(f"样本量: {len(samples_df)}")
        return 0

    if args.command == "train":
        cfg = TrainConfig(
            events_root=args.events_root,
            market_file=args.market_file,
            out_dir=args.out_dir,
            max_features=args.max_features,
            window_bars=args.window_bars,
            future_bars=args.future_bars,
            train_ratio=args.train_ratio,
            valid_ratio=args.valid_ratio,
            min_year=args.min_year,
        )
        train_pipeline(cfg)
        return 0

    if args.command == "predict":
        predict_one(
            bundle_path=args.bundle,
            market_file=args.market_file,
            event_text=args.event_text,
            publish_time=args.publish_time,
            event_type=args.event_type,
            model_name=args.model_name,
            out_json=args.out_json,
        )
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
