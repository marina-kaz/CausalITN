from tap import Tap
from time import perf_counter
from utils.constants import TEST_DATA_PATH
from inference.inference_utils import save_report, get_test_turns, compose_modelfile, create_model
import requests
import json


class InferenceCLI(Tap):
    model: str = 'mistral'
    temperature: float = 0.0
    shots: int = 3
    verbose: bool = True
    force_recollect: bool = False


def setup_ollama(model: str, temperature: float, shots: int) -> str:
    modelfile = compose_modelfile(model=model, temperature=temperature, shots=shots)
    model_name = create_model(modelfile_path=modelfile)
    return model_name


def get_response(model: str, messages: list[dict[str, str]]):
    r = requests.post(
        "http://0.0.0.0:11434/api/chat",
        json={"model": model, "messages": messages, "stream": True},
    )
    r.raise_for_status()

    output = ""

    for line in r.iter_lines():
        body = json.loads(line)
        if "error" in body:
            raise Exception(body["error"])
        if body.get("done") is False:
            message = body.get("message", "")
            content = message.get("content", "")
            output += content

        if body.get("done", False):
            return output


def collect_predictions(model: str, verbose=False):
    results = []
    previous_category = None
    for source, target, global_id, category in get_test_turns(test_dir=TEST_DATA_PATH):
        start = perf_counter()
        try:
            response = get_response(model=model,
                                    messages=[{"role": "user", 
                                               "content": f"Денормализуй: {source}"}])
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
            name = model.split('_')[0]
            t = model.split('_')[1][1:]
            r = model.split('_')[2][1:]
            save_report(report=[result for result in results if result['category'] == previous_category],
                        model=name,
                        temperature=t,
                        shots=r)
        previous_category = category
        # ############

        print(f'Source:\n\t{source}\nResponse:\n\t{response}\nTarget:\n\t{target}\nCategory\n\t{category}')

    return results


def main() -> None:
    args = InferenceCLI(underscores_to_dashes=True).parse_args()
    oll_model = setup_ollama(model=args.model, temperature=args.temperature, shots=args.shots)
    global_start = perf_counter()
    results = collect_predictions(model=oll_model,
                                  verbose=args.verbose)
    global_finish = perf_counter()
    print('Total Elapsed time:', global_finish - global_start)
    save_report(report=results, 
                model=args.model, 
                temperature=args.temperature, 
                shots=args.shots)


if __name__ == "__main__":
    main()
