from openai import AzureOpenAI
from pathlib import Path
import pandas as pd
from utils.constants import REPORT_PATH, PROMPT_PATH
import subprocess
from utils.utils import get_system_prompt



def get_azure_client():
    client = AzureOpenAI(
    api_key="",  
    api_version="2023-03-15-preview",
    azure_endpoint="https://sashagpt4turbo.openai.azure.com/"
    )
    return client


def get_fake_turns(prompt_dir: Path, shots: int) -> tuple[list[dict[str, str]], list[int]]:
    global_indices = []
    messages = []
    for prompt_path in prompt_dir.iterdir():
        if not prompt_path.name.endswith('.csv'):
            continue
        prompt_data = pd.read_csv(prompt_path, sep='\t')
        prompt_data.dropna(inplace=True)
        promt_sample = prompt_data.sample(n=shots, random_state=10)
        global_indices.extend(promt_sample.global_id.tolist())

        for user, system in zip(promt_sample.normalized.to_dict().values(),
                                promt_sample.denormalized.to_dict().values()):
            messages.extend([
                {'role': 'user', 'content': f'Денормализуй: {user}'},
                {'role': 'assistant', 'content': system}
            ])
    return messages, global_indices


def get_test_turns(test_dir: Path):
    for test_path in test_dir.iterdir():
        test_data = pd.read_csv(test_path, sep='\t')
        test_data.dropna(inplace=True)
        test_data.global_id.astype(int, copy=False, errors='ignore')
        category = test_path.name.split('_')[0]

        for source, target, glob_id in zip(
            test_data.normalized.values,
            test_data.denormalized.values,
            test_data.global_id.values,
        ):
            yield source, target, glob_id, category


def save_report(report: dict, model: str, temperature: float, shots: int):
    # соломка
    data = pd.DataFrame(report)
    data.to_csv(f'backup/t{temperature}_r{shots}.csv', sep='\t', index=False)

    predictions_path = REPORT_PATH / model /  'predictions'
    if not predictions_path.exists():
        predictions_path.mkdir(parents=True)

    data = pd.DataFrame(report)
    for category in data.category.unique():
        cat_sample = data[data.category == category]
        cat_sample.to_csv(REPORT_PATH / model /  'predictions' / f't{temperature}_r{shots}_{category}.csv', sep='\t', index=False)
    

def compose_modelfile(model: str, temperature: float, shots: int):
    modelfile_source_path = PROMPT_PATH / 'modelfiles' / 'base' / f'{model}_base'
    modelfile_target_path = PROMPT_PATH / 'modelfiles' / f'{model}_t{temperature}_r{shots}'

    with open(modelfile_source_path, 'r', encoding='utf-8') as modelfile_source_file:
        modelfile = modelfile_source_file.read()

    modelfile += f'PARAMETER temperature {temperature}\n'
    modelfile += f'SYSTEM "{get_system_prompt()}"\n'
    for turn in [f'MESSAGE {turn["role"].replace("system", "assistant" )} {turn["content"]}'
                 for turn in get_fake_turns(prompt_dir=PROMPT_PATH, shots=shots)[0]]:
        modelfile += f'{turn}\n'

    with open(modelfile_target_path, 'w', encoding='utf-8') as modelfile_target_file:
        modelfile_target_file.write(modelfile)

    return modelfile_target_path


def create_model(modelfile_path: Path) -> str:
    model_name = modelfile_path.name
    command = f"ollama create {model_name} -f {modelfile_path}"
    print(command)
    process = subprocess.run(command, shell=True)
    print('Return code:', process.returncode)
    return model_name
