# CausalITN

Course paper repository for Inverse Text Normalization (the Russian language) via Large Language Models.

## Quickstart 

0. Prerequisities

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements
```

1. Collect predictions for models

```bash
python inference/parameter_search.py --model gpt-4o --kind azure
```

Make sure to specify your API token!


2. Calculate metrics

```bash
python evaluate/calculate_metrics.py --model gpt-4o
```

3. Format table
```bash
python evaluate/format_parameter_table.py --model gpt-4o
```
