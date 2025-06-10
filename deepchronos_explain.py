#!/usr/bin/env python3
"""
DeepChronos First-Iteration: Explain with ChronosFineTuned[bolt_small]

Pipeline:
  1. Reads tasks from `tasks.yaml`.
  2. Standardizes and validates time-series data.
  3. Trains ChronosFineTuned[bolt_small] via AutoGluon.
  4. Builds a prompt & queries DeepSeek-R1 for explanations.
  5. Prints prompts and explanations.
"""

import os
import time
import logging
import yaml
import pandas as pd
import numpy as np
import requests
import subprocess
import pkg_resources

from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame
import fev  # pip install git+https://github.com/autogluon/fev.git@v0.3.1

# --------------------------
# Logging Configuration
# --------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("DeepChronos")

# --------------------------
# Load Configs & Tasks
# --------------------------
def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

conf = load_yaml("config.yaml")
tasks = load_yaml("tasks.yaml")

# --------------------------
# Data Preparation Utils
# --------------------------
def standardize_columns(df):
    df.columns = [c.lower() for c in df.columns]
    return df

def ensure_item_id(df):
    if 'item_id' not in df.columns:
        df['item_id'] = 1
    return df

def convert_timestamp(df):
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df.dropna(subset=['timestamp'], inplace=True)
    return df

def convert_target(df, target_col):
    df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
    return df

def prepare_dataframe(df: pd.DataFrame, target_col: str):
    df = standardize_columns(df)
    df = ensure_item_id(df)
    df = convert_timestamp(df)
    # rename alternative target columns to 'target'
    if target_col not in df.columns:
        for alt in ['ot', 'y']:
            if alt in df.columns:
                logger.info(f"Renaming '{alt}' to {target_col}")
                df.rename(columns={alt: target_col}, inplace=True)
                break
    if target_col not in df.columns:
        raise ValueError(f"Missing target column '{target_col}'")
    df = convert_target(df, target_col)
    df.rename(columns={target_col: 'target'}, inplace=True)
    return df

def derive_frequency(seasonality: int) -> str:
    if seasonality in [24,48]: return 'H'
    if seasonality==12: return 'M'
    if seasonality==4: return 'Q'
    if seasonality in [7,52]: return 'W'
    return 'D'

# --------------------------
# DeepSeek Server Management
# --------------------------
def start_deepseek_server():
    model = conf['deepseek']['model']
    dtype = conf['deepseek']['dtype']
    cmd = f"CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model {model} --dtype={dtype}"
    env = os.environ.copy()
    process = subprocess.Popen(cmd, shell=True, env=env)
    logger.info(f"Starting DeepSeek-R1 (PID {process.pid})")
    time.sleep(10)  # wait for server
    return process

def query_deepseek(prompt):
    url = conf['deepseek']['url']
    try:
        res = requests.post(url, json={'prompts': [prompt]}, timeout=60)
        res.raise_for_status()
        data = res.json()
        return data.get('responses', [data.get('text', '')])[0]
    except Exception as e:
        logger.error(f"DeepSeek query failed: {e}")
        return f"Error: {e}"

# --------------------------
# Main Pipeline
# --------------------------
def run_pipeline():
    # FEv version check
    try:
        v = pkg_resources.get_distribution("fev").version
        if tuple(map(int, v.split('.'))) < (0,3,1):
            logger.warning(f"fev version {v} < 0.3.1; proceed with caution")
    except:
        logger.warning("Could not verify fev version")

    results = {}
    for cfg in tasks:
        name = cfg['dataset_config']
        logger.info(f"Processing task: {name}")
        try:
            task = fev.Task(
                dataset_path=cfg['dataset_path'],
                dataset_config=name,
                horizon=cfg['horizon']
            )
            df_train, df_test, _ = fev.convert_input_data(task, adapter='pandas')
            df_train = prepare_dataframe(df_train, cfg.get('target_col','target'))
            df_test  = prepare_dataframe(df_test,  cfg.get('target_col','target'))

            train_ts = TimeSeriesDataFrame(df_train)
            test_ts  = TimeSeriesDataFrame(df_test)

            freq = derive_frequency(cfg['seasonality'])
            pred_len = cfg['horizon']

            predictor = TimeSeriesPredictor(
                target='target',
                prediction_length=pred_len,
                freq=freq,
                eval_metric=conf['autogluon']['eval_metric']
            )
            predictor.fit(train_ts, hyperparameters={conf['autogluon']['chronos_variant']: {}})

            fc = predictor.predict(test_ts).rename(columns={'mean':'forecast'}, errors='ignore')
            tsid = fc.iloc[0]['item_id']
            forecast_vals = (
                fc[fc['item_id']==tsid]
                .sort_values('timestamp')['forecast']
                .tolist()
            )
            last_obs = (
                df_train[df_train['item_id']==tsid]
                .sort_values('timestamp')['target']
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
            explanation = query_deepseek(prompt)
            results[name] = {'prompt': prompt, 'explanation': explanation}

        except Exception as e:
            logger.error(f"Task {name} failed: {e}")
            results[name] = {'error': str(e)}

    return results

# --------------------------
# Execute
# --------------------------
if __name__ == '__main__':
    ds_proc = start_deepseek_server()
    try:
        out = run_pipeline()
        print("\n=== Results ===")
        for name, res in out.items():
            print(f"\n-- {name} --")
            if 'error' in res:
                print("ERROR:", res['error'])
            else:
                print("PROMPT:\n", res['prompt'])
                print("EXPLANATION:\n", res['explanation'])
    finally:
        ds_proc.terminate()
        logger.info("DeepSeek-R1 server terminated.")
