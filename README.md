# DeepChronos

DeepChronos merges AutoGluon's Chronos foundation model with DeepSeek-R1 LLM reasoning to produce explainable time-series forecasts.  

## Features
- Data ingestion via FEV (Hugging Face)
- Automatic column standardization & validation
- ChronosFineTuned[bolt_small] training
- Prompt construction & narrative explanation via DeepSeek-R1

## Requirements
- Python 3.8+
- CUDA (if serving DeepSeek-R1 with GPU)

## Installation
```bash
git clone https://github.com/yourusername/DeepChronos.git
cd DeepChronos
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
