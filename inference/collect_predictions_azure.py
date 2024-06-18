from tap import Tap
from time import perf_counter
from utils.utils import get_system_prompt
from utils.constants import PROMPT_PATH, TEST_DATA_PATH, REPORT_PATH
from inference.inference_utils import save_report, get_test_turns, get_fake_turns, get_azure_client


class InferenceCLI(Tap):
    model: str = 'gpt-4-1106-preview'
    temperature: float = 0.5
    shots: int = 2
    verbose: bool = False
    force_recollect: bool = False


def get_response(client, system_prompt, fake_turns, test_prompt, engine: str, temperature: int):

    response = client.chat.completions.create(
        model=engine,
        messages=[
            {"role": "system", "content": system_prompt},
            *fake_turns,
            {"role": "user", "content": test_prompt}
        ],
        temperature=temperature,
        n=1
    )
    return response.choices[0].message.content


def collect_predictions(client, system_prompt, fake_turns, temperature, verbose=False):
    results = []
    previous_category = None
    for source, target, global_id, category in get_test_turns(test_dir=TEST_DATA_PATH):
        start = perf_counter()
        try:
            response = get_response(client=client,
                                    system_prompt=system_prompt,
                                    engine='gpt-4-fast',
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
                        model='gpt-4-1106-preview',
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
    client = get_azure_client()
    system_promts = get_system_prompt()
    results = collect_predictions(client=client, 
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
