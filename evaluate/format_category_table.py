from pathlib import Path 
from utils.constants import REPORT_PATH
from utils.utils import load_report, get_best_params
import pandas as pd 


def get_models() -> list[str]:
    return [model.name for model in REPORT_PATH.iterdir() if model.is_dir()]


def collect_category_metrics(model: str, param: str):
    report = load_report(model=model)
    results = {}
    for category, details in report[param].items():
        if category == 'global_metrics':
            results[category] = details['acc']
        else:
            results[category] = details['metrics']['acc']
    return results


def collect_scores(models: list[str]) -> dict[str, list[float]]:
    scores = []
    for model in models:
        best_param = get_best_params(model)
        best_param = 't0.25_r3'
        # print(best_param)
        category_metrics = collect_category_metrics(model=model, param=str(best_param))
        scores.append(category_metrics)
    return {k: [dic[k] for dic in scores] for k in scores[0]}


def save_csv_table(report: pd.DataFrame) -> None:
    report.to_csv(REPORT_PATH / 'category_table.csv', index=True)


def save_latex_table(report: pd.DataFrame) -> None:
    latex_str = report.style.format({'gpt-4-1106-preview': "{:.2f}"}).to_latex(
        column_format='lc', 
        hrules=True,
        position_float='centering',
        multirow_align='c',
        multicol_align='r'
    )
    with open(REPORT_PATH / 'category_table.tex', 'w') as f:
        f.write(latex_str)

models = get_models()
scores = collect_scores(models=models)
data = pd.DataFrame(scores, index=models).transpose().rename_axis('Category')
save_csv_table(data)
save_latex_table(data)

print(data)
