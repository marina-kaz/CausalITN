
import json 
from evaluate.dto.report import Parameters
from utils.constants import REPORT_PATH, PROMPT_PATH

import re 

def get_system_prompt() -> str:
    with open(PROMPT_PATH / 'system_prompt.txt', 'r', encoding='utf-8') as spfile:
            system_prompt = spfile.read()
    return system_prompt


def load_report(model: str) -> dict:
    if not (REPORT_PATH / model).exists():
        raise ValueError(f'No predictions for model {model} at {REPORT_PATH / model}')
    if not (REPORT_PATH / model / 'results' / 'metrics.json').exists():
        raise ValueError(f'Model {model} is not evaluated!')
    with open(REPORT_PATH / model / 'results' / 'metrics.json') as rfile:
        report = json.load(rfile)
    return report


def get_params(report_name: str) -> Parameters:
    return Parameters(
        temperature=re.search(r't(-?\d+(\.\d+)?)', report_name).group(1),
        shots=re.search(r'r(\d+)', report_name).group(1)
    )


def get_eval_params(model: str) -> list[str]:
    report = load_report(model=model)
    return list(report.keys())


def get_best_params(model: str) -> str:
    report = load_report(model=model)
    best_param = None
    best_acc = 0
    for parameters in report:
        acc = report[parameters]['global_metrics']['acc']
        if acc > best_acc:
            best_acc = acc
            best_param = parameters
    return get_params(best_param)


def get_global_acc(model: str, param: str) -> str | None:
    try:
        return load_report(model=model)[param]['global_metrics']['acc']
    except (ValueError, KeyError):
        print(f'No report for {model} with params {param}')
        return None
