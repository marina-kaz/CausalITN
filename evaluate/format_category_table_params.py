from pathlib import Path 
from utils.constants import REPORT_PATH
from utils.utils import load_report, get_best_params, get_eval_params
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
    headers = []
    for model in models:
        params = get_eval_params(model=model)
        for param in params:
            category_metrics = collect_category_metrics(model=model, param=str(param))
            scores.append(category_metrics)
            headers.append(f'{model}_{param}')
    data = pd.DataFrame(scores).transpose().rename_axis('Category')
    # print(data)
    data.columns = headers
    return data


def save_csv_table(report: pd.DataFrame) -> None:
    report.to_csv(REPORT_PATH / 'category_table_params.csv', index=True)
    report.transpose().round(3).sort_index().to_excel(REPORT_PATH / 'category_table_params.xlsx', index=True)



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
save_csv_table(scores)
print(scores.transpose().round(3).sort_index())
