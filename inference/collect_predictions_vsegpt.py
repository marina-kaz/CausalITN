from tap import Tap
from time import perf_counter
from utils.utils import get_system_prompt
from utils.constants import PROMPT_PATH, TEST_DATA_PATH
from inference.inference_utils import save_report, get_test_turns, get_fake_turns
import openai


FULL_NAMES = {
    'saiga-mistral-7b': 'gusev/saiga-mistral-7b',
    'claude-3-haiku': 'anthropic/claude-3-haiku',
    'claude-3-sonnet': 'anthropic/claude-3-sonnet',
    'claude-3-opus': 'anthropic/claude-3-opus',
    'gpt-3.5-turbo-instruct': 'openai/gpt-3.5-turbo-instruct',
    'gpt-3.5-turbo-1106': 'openai/gpt-3.5-turbo-1106',
    'gpt-4o': 'openai/gpt-4o',
    'mixtral-8x7b-instruct': 'mistralai/mixtral-8x7b-instruct',
    'gemini-flash-1.5': 'google/gemini-flash-1.5',
    'gemini-pro': 'google/gemini-pro'
}


class InferenceCLI(Tap):
    model: str = 'claude-3-haiku'
    temperature: float = 0.0
    shots: int = 3
    verbose: bool = True
    force_recollect: bool = False


def setup_vsegpt() -> None:
    openai.api_key = ""
    openai.base_url = "https://api.vsegpt.ru/v1/"


def get_response(model: str, system_prompt, fake_turns, test_prompt, temperature: float):
    response = openai.chat.completions.create(
        model=FULL_NAMES[model],
        messages=[
            {"role": "system", "content": system_prompt},
            *fake_turns,
            {"role": "user", "content": test_prompt}
        ],
        temperature=temperature,
        n=1,
        max_tokens=3000, 
    )
    return response.choices[0].message.content


def collect_predictions(model, system_prompt, fake_turns, temperature, verbose=False):
    results = []
    previous_category = None
    for source, target, global_id, category in get_test_turns(test_dir=TEST_DATA_PATH):
        start = perf_counter()
        try:
            response = get_response(model=model,
                                    system_prompt=system_prompt,
                                    fake_turns=fake_turns,
                                    test_prompt=f'Денормализуй: {source}',
                                    temperature=temperature)
        except Exception as exception:
            response = 'error occurred'
            print(f'Exception occurred with {source}: {exception}')
        finish = perf_counter()
        
        result = {
            'source': source,
            'response': response,
            'target': target,
            'global_id': global_id,
            'elapsed_time': finish - start,
            'category': category
        }
        results.append(result)

        # ##########
        if category != previous_category and previous_category is not None:
            save_report(report=[result for result in results if result['category'] == previous_category],
                        model=model,
                        temperature=temperature,
                        shots=int(len(fake_turns) / 10 / 2))
        previous_category = category
        # ############

        if verbose:
            print(f'Source:\n\t{source}\nResponse:\n\t{response}\nTarget:\n\t{target}\nCategory\n\t{category}')

    return results


def main() -> None:
    args = InferenceCLI(underscores_to_dashes=True).parse_args()
    fake_turns, _ = get_fake_turns(prompt_dir=PROMPT_PATH, shots=args.shots)
    global_start = perf_counter()
    system_promts = get_system_prompt()
    setup_vsegpt()
    results = collect_predictions(model=args.model, 
                                  system_prompt=system_promts, 
                                  fake_turns=fake_turns,
                                  temperature=args.temperature,
                                  verbose=args.verbose)
    global_finish = perf_counter()
    print('Total Elapsed time:', global_finish - global_start)
    save_report(report=results,
                model=args.model,
                temperature=args.temperature, 
                shots=args.shots)


if __name__ == "__main__":
    main()
