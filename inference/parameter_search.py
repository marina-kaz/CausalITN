
import subprocess
from utils.utils import get_global_acc, get_best_params
from tap import Tap
from utils.constants import REPORT_PATH
from pathlib import Path
import json
import re

class InferenceCLI(Tap):
    model: str = 'llama3'
    kind: str = 'ollama'
    temperature_grid: list[float] = [0.0, 0.5, 1.0]
    shots_grid: list[int] = [1, 2, 3]
    default_shots: int = 3

    def process_args(self):
        if self.kind not in ['azure', 'ollama', 'vsegpt']:
            raise ValueError(f'Kind argument {self.kind} does not conform')


def create_report(model: str) -> Path:
    dir_path = REPORT_PATH / model / 'results'
    if not dir_path.exists():
        dir_path.mkdir(parents=True)
    file_path = dir_path / 'param_search.json'
    if not file_path.exists():
        with open(file_path, 'w', encoding='utf-8') as rfile:
            json.dump({}, rfile)
    return file_path


def update_report(report_path: Path, param: str, acc: float) -> None:
    with open(report_path, 'r', encoding='utf-8') as rfile:
        report = json.load(rfile)
    report[param] = acc
    with open(report_path, 'w', encoding='utf-8') as rfile:
        json.dump(report, rfile)


def collect_predictions(temperature: float, shots: int, model: str, kind: str):

    script_kinds = {
        'azure': 'inference/collect_predictions_azure.py',
        'ollama': 'inference/collect_predictions_ollama.py',
        'vsegpt': 'inference/collect_predictions_vsegpt.py',
    }
    log_file = f'logs/t{temperature}_r{shots}_{model}_log.txt'
    command = f"python {script_kinds[kind]} --model {model} --temperature {temperature} --shots {shots} --verbose > {log_file}"
    print(command)
    process = subprocess.run(command, shell=True)
    print('Return code:', process.returncode)

def calculate_metrics(model: str):
    command = f"python evaluate/calculate_metrics.py --model {model}"
    print(command)
    process = subprocess.run(command, shell=True)
    print('Return code:', process.returncode)


def main() -> None:
    args = InferenceCLI(underscores_to_dashes=True).parse_args()

    print(args.temperature_grid)
    model = InferenceCLI().parse_args().model
    report_path = create_report(model=model)

    for temperature in args.temperature_grid:
        for shots in args.shots_grid:
            accuracy = get_global_acc(model=model, param=f't{temperature}_r{shots}')
            if accuracy is None:
                print(f'No measurements were made for temp {temperature} and shots {shots}!')
                collect_predictions(temperature=temperature, shots=shots, model=model, kind=args.kind)
                calculate_metrics(model=model)
                accuracy = get_global_acc(model=model, param=f't{temperature}_r{shots}')
                assert accuracy is not None
                print(f'Obtained accuracy {accuracy} for temp {temperature} and shots {shots}!')
            update_report(report_path=report_path, param=f't{temperature}_r{shots}', acc=accuracy)

if __name__ == "__main__":
    main()
