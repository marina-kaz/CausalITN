import re

from tap import Tap
import pandas as pd

from pathlib import Path
import json

from utils.constants import REPORT_PATH, TEST_DATA_PATH
from evaluate.dto.report import Parameters
from utils.categories import Category
import jiwer

class EvaluationCLI(Tap):

    model: str = 'gpt-4-1106-preview'


def get_params(report_name: str) -> Parameters:
    return Parameters(
        temperature=re.search(r't(-?\d+(\.\d+)?)', report_name).group(1),
        shots=re.search(r'r(\d+)', report_name).group(1)
    )

def get_category(report_name: str) -> Category:
    # return Category(report_name.split('_')[-1])
    return report_name.split('_')[-1]



def load_test_data() -> dict[Category, pd.DataFrame]:
    test_data = {}
    for test_file in TEST_DATA_PATH.iterdir():
        data = pd.read_csv(test_file, sep='\t')
        category = test_file.stem.split('_')[0]
        test_data[category] = data
    return test_data

def load_predictions(pred_dir: Path) -> dict[Parameters: dict[Category, pd.DataFrame]]:
    predictions = {}
    for response_file in (pred_dir).iterdir():
        params = str(get_params(response_file.name))
        if not params in predictions:
            predictions[params] = {}
        category = get_category(response_file.stem)
        predictions[params][category] = pd.read_csv(response_file, sep='\t')
    return predictions


def calculate_metrics(data: pd.DataFrame) -> dict[str, float]:
    # data['denormalized'] = data.deno
    data['wer'] = data.apply(lambda x: jiwer.wer(str(x.denormalized), str(x.response)), axis=1)
    data['cer'] = data.apply(lambda x: jiwer.cer(str(x.denormalized), str(x.response)), axis=1)
    data['acc'] = data.apply(lambda x: str(x.denormalized) == str(x.response), axis=1)

    category_metrics = {
        'wer': data.wer.mean(),
        'cer': data.cer.mean(),
        'acc': data.acc.mean(),
    }
    return category_metrics


def assemble_detailed_report(data: pd.DataFrame) -> dict[str, str | dict[str, float]]:
    details = {}
    for sample in data.to_dict('records'):
        details[sample['global_id']] = {
            'source': sample['normalized'],
            'target': sample['denormalized'],
            'prediction': sample['response'],
            'metrics': {
                'wer': sample['wer'],
                'cer': sample['cer'],
                'acc': int(sample['acc']),
            }
        }
    return details


def calcualte_global_metrics(param_dict: dict) -> dict:
    total_scores = {'wer': 0, 'cer': 0, 'acc': 0}
    total_predicted = 0
    for category in param_dict.keys():
        for sample in param_dict[category]['details'].values():
            total_scores['wer'] += sample['metrics']['wer']
            total_scores['cer'] += sample['metrics']['cer']
            total_scores['acc'] += sample['metrics']['acc']
            total_predicted += 1

    global_metrics = {metric: score / total_predicted for metric, score in total_scores.items()}
    return global_metrics


def evaluate(test_data: pd.DataFrame, pred_data: pd.DataFrame) -> dict:
    results = {}
    for parameters, categories in pred_data.items():
        if not parameters in results:
            results[parameters] = {}
        for category, predicted in categories.items():
            results[parameters][category] = {}
            data = pd.merge(
                test_data[category][['global_id', 'normalized', 'denormalized']], 
                predicted[['global_id', 'response']], on='global_id', how='outer')
            data.dropna(inplace=True)
            results[parameters][category]['metrics'] = calculate_metrics(data)
            results[parameters][category]['details'] = assemble_detailed_report(data)
        results[parameters]['global_metrics'] = calcualte_global_metrics(results[parameters])
    return results


def dump_metrics(report: dict, model: str) -> None:
    with open(REPORT_PATH / model / 'results' / 'metrics.json', 'w', encoding='utf-8') as rfile:
        json.dump(report, rfile, ensure_ascii=False, indent=4)



def main() -> None:
    args = EvaluationCLI().parse_args()
    predictions = load_predictions(REPORT_PATH / args.model / 'predictions')
    test_data = load_test_data()
    report = evaluate(test_data=test_data, pred_data=predictions)
    dump_metrics(report=report, model=args.model)


if __name__ == "__main__":
    main()
