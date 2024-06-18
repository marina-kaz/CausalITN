from tap import Tap

import random
from utils.utils import load_report
from utils.utils import get_best_params



class AnalyzerCLI(Tap):

    model: str = 'neural-chat'
    params: str | None = 't0.5_r3' 
    category: str = 'FRACTION'
    n: int = 10
    seed: int = 10


def show_samples(samples: dict, n: int):
    
    errors = [(index, sample) for index, sample
            in samples['details'].items() 
            if not sample['metrics']['acc']]
    print(f'Total errors: {len(errors)}/{len(samples["details"])}')
    if n < len(samples):
        errors = random.sample(errors, n)
    for index, sample in errors:
        print(f'Index:\n\t{index}\n')
        print(f'Source:\n\t{sample["source"]}\n')
        print(f'Target:\n\t{sample["target"]}\n')
        print(f'Prediction:\n\t{sample["prediction"]}\n')
        print()



def main() -> None:
    args = AnalyzerCLI().parse_args()
    random.seed(args.seed)
    report = load_report(model=args.model)
    if not args.params:
        params = get_best_params(model=args.model)
    else:
        params = args.params
    samples = report[str(params)][args.category]
    show_samples(samples=samples, n=args.n)


if __name__ == "__main__":
    main()