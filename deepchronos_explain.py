#!/usr/bin/env python3
"""DeepChronos pipeline for training Chronos models and generating
explanations with DeepSeek-R1."""

from __future__ import annotations

import logging
import os
import time
import subprocess
from dataclasses import dataclass
from typing import Dict, List

import yaml
import pandas as pd
import requests
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame
import fev  # pip install git+https://github.com/autogluon/fev.git@v0.3.1
import pkg_resources


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------
@dataclass
class DeepSeekConfig:
    model: str
    dtype: str
    url: str
    device: str = "0"


@dataclass
class AutoGluonConfig:
    eval_metric: str
    chronos_variant: str


@dataclass
class TaskConfig:
    dataset_path: str
    dataset_config: str
    horizon: int
    seasonality: int
    target_col: str = "target"


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("DeepChronos")


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
def load_yaml(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.lower() for c in df.columns]
    return df


def ensure_item_id(df: pd.DataFrame) -> pd.DataFrame:
    if "item_id" not in df.columns:
        df["item_id"] = 1
    return df


def convert_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df.dropna(subset=["timestamp"], inplace=True)
    return df


def convert_target(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    return df


def prepare_dataframe(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    df = standardize_columns(df)
    df = ensure_item_id(df)
    df = convert_timestamp(df)

    if target_col not in df.columns:
        for alt in ("ot", "y"):
            if alt in df.columns:
                logger.info("Renaming '%s' to %s", alt, target_col)
                df.rename(columns={alt: target_col}, inplace=True)
                break
    if target_col not in df.columns:
        raise ValueError(f"Missing target column '{target_col}'")

    df = convert_target(df, target_col)
    df.rename(columns={target_col: "target"}, inplace=True)
    return df


FREQ_MAP = {24: "H", 48: "H", 12: "M", 4: "Q", 7: "W", 52: "W"}


def derive_frequency(seasonality: int) -> str:
    return FREQ_MAP.get(seasonality, "D")


# ---------------------------------------------------------------------------
# DeepSeek server management
# ---------------------------------------------------------------------------
class DeepSeekServer:
    def __init__(self, cfg: DeepSeekConfig):
        self.cfg = cfg
        self.process: subprocess.Popen | None = None

    def __enter__(self) -> "DeepSeekServer":
        cmd = (
            f"CUDA_VISIBLE_DEVICES={self.cfg.device} trl vllm-serve --model"
            f" {self.cfg.model} --dtype={self.cfg.dtype}"
        )
        self.process = subprocess.Popen(cmd, shell=True, env=os.environ.copy())
        logger.info("Starting DeepSeek-R1 (PID %s)", self.process.pid)
        time.sleep(10)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.process:
            self.process.terminate()
            logger.info("DeepSeek-R1 server terminated.")


class DeepSeekClient:
    def __init__(self, url: str):
        self.url = url
        self.session = requests.Session()

    def query(self, prompt: str) -> str:
        try:
            res = self.session.post(self.url, json={"prompts": [prompt]}, timeout=60)
            res.raise_for_status()
            data = res.json()
            return data.get("responses", [data.get("text", "")])[0]
        except Exception as e:  # noqa: BLE001
            logger.error("DeepSeek query failed: %s", e)
            return f"Error: {e}"


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(tasks: List[TaskConfig],
                 ds_client: DeepSeekClient,
                 ag_conf: AutoGluonConfig) -> Dict[str, Dict]:
    try:
        v = pkg_resources.get_distribution("fev").version
        if tuple(map(int, v.split("."))) < (0, 3, 1):
            logger.warning("fev version %s < 0.3.1; proceed with caution", v)
    except Exception:  # noqa: BLE001
        logger.warning("Could not verify fev version")

    results: Dict[str, Dict] = {}
    for cfg in tasks:
        name = cfg.dataset_config
        logger.info("Processing task: %s", name)
        try:
            task = fev.Task(
                dataset_path=cfg.dataset_path,
                dataset_config=cfg.dataset_config,
                horizon=cfg.horizon,
            )
            df_train, df_test, _ = fev.convert_input_data(task, adapter="pandas")
            df_train = prepare_dataframe(df_train, cfg.target_col)
            df_test = prepare_dataframe(df_test, cfg.target_col)

            train_ts = TimeSeriesDataFrame(df_train)
            test_ts = TimeSeriesDataFrame(df_test)

            freq = derive_frequency(cfg.seasonality)
            pred_len = cfg.horizon

            predictor = TimeSeriesPredictor(
                target="target",
                prediction_length=pred_len,
                freq=freq,
                eval_metric=ag_conf.eval_metric,
            )
            predictor.fit(train_ts, hyperparameters={ag_conf.chronos_variant: {}})

            fc = predictor.predict(test_ts).rename(columns={"mean": "forecast"}, errors="ignore")
            tsid = fc.iloc[0]["item_id"]
            forecast_vals = (
                fc[fc["item_id"] == tsid]
                .sort_values("timestamp")["forecast"]
                .tolist()
            )
            last_obs = (
                df_train[df_train["item_id"] == tsid]
                .sort_values("timestamp")["target"]
                .tail(pred_len)
                .tolist()
            )

            prompt = (
                f"TS ID: {tsid}\n"
                f"Freq: {freq}, Horizon: {pred_len}\n"
                f"Last Observed: {last_obs}\n"
                f"Forecast: {forecast_vals}\n\n"
                "Explain the likely reasoning behind this forecast."
            )
            explanation = ds_client.query(prompt)
            results[name] = {"prompt": prompt, "explanation": explanation}
        except Exception as e:  # noqa: BLE001
            logger.error("Task %s failed: %s", name, e)
            results[name] = {"error": str(e)}
    return results


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    conf = load_yaml("config.yaml")
    tasks_yaml = load_yaml("tasks.yaml")

    ds_cfg = DeepSeekConfig(**conf["deepseek"])
    ag_cfg = AutoGluonConfig(**conf["autogluon"])
    tasks_list = [TaskConfig(**t) for t in tasks_yaml]

    ds_client = DeepSeekClient(ds_cfg.url)

    with DeepSeekServer(ds_cfg):
        output = run_pipeline(tasks_list, ds_client, ag_cfg)
        print("\n=== Results ===")
        for name, res in output.items():
            print(f"\n-- {name} --")
            if "error" in res:
                print("ERROR:", res["error"])
            else:
                print("PROMPT:\n", res["prompt"])
                print("EXPLANATION:\n", res["explanation"])
