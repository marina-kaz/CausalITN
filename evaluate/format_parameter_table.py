import re

from tap import Tap
import pandas as pd
import json

from utils.constants import REPORT_PATH
from utils.utils import load_report

class ParamFormatterCLI(Tap):

    model: str = 'gpt-4-1106-preview'
    """
    Model to evaluate
    """

def collect_metrics(report: dict) -> pd.DataFrame:
    metrics = []
    for parameters in report:
        metrics.append(
            {
                'R': re.search(r'r(\d+)', parameters).group(1),
                'T': re.search(r't(-?\d+(\.\d+)?)', parameters).group(1),
                'Accuracy': report[parameters]['global_metrics']['acc'],
                'WER': report[parameters]['global_metrics']['wer'],
                'CER': report[parameters]['global_metrics']['cer'],
            }
        )
    return pd.DataFrame(metrics)


def save_csv_table(report: pd.DataFrame, model: str) -> None:
    report.to_csv(REPORT_PATH / model / 'results' / 'param_table.csv', index=False)


def highlight_best(s):
    if s.name == 'Accuracy':
        is_best = s == s.max()
    else:
        is_best = s == s.min()
    return ['\\textbf{' + '{:.2f}'.format(v) + '}' if is_best.iloc[i] else '{:.2f}'.format(v) for i, v in enumerate(s)]


def save_latex_table(report: pd.DataFrame, model: str) -> None:
    report[['Accuracy', 'WER', 'CER']] = report[['Accuracy', 'WER', 'CER']].apply(highlight_best)
    latex_str = report.to_latex(index=False, escape=False, column_format='ccccc')
    latex_str = latex_str.replace('\\textbf', '\\bf')
    latex_str = latex_str.replace('\\toprule', '\\hline')
    latex_str = latex_str.replace('\\midrule', '\\hline')
    latex_str = latex_str.replace('\\bottomrule', '\\hline')
    with open(REPORT_PATH / model / 'results' / 'param_table.tex', 'w') as f:
        f.write(latex_str)

def main() -> None:
    args = ParamFormatterCLI().parse_args()
    report = load_report(model=args.model)
    metrics = collect_metrics(report=report)
    save_csv_table(report=metrics, model=args.model)
    print(metrics)
    save_latex_table(report=metrics, model=args.model)


if __name__ == "__main__":
    main()
