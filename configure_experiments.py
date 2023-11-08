"""Create a separate file with a list of experiments and their Slurm task id's"""
import argparse
import logging
import pandas as pd
import traceback

from datetime import datetime
from pathlib import Path
from typing import Any

from benchmark import task_descriptions


bf_experiments = [
    {'problem': problem,
     'language': language,
     'branching_factor': branching_factor,
     'max_programs': 100,
     'beam_width': branching_factor,
     'debug_prompt_id': 0,
     'log': 'INFO'}
    for branching_factor in (2, 4, 16, 1, 10, 100)
    for problem in task_descriptions.keys()
    for language in ('C++', 'Python')
]

experiments = bf_experiments


def update_experiments_list(input_file: Path | str, experiments: list[dict[str, Any]]):
    """Append a new set of hyperparameters from `experiments` list
     to the previous experiments taken from `input_file`"""
    new_experiments = pd.DataFrame(experiments)
    try:
        previous_experiments = pd.read_csv(input_file, header=0, index_col=0)
        updated_experiments = pd.concat((previous_experiments, new_experiments), ignore_index=True)
    except FileNotFoundError:
        updated_experiments = new_experiments
    updated_experiments.index = list(range(1, updated_experiments.shape[0] + 1))
    updated_experiments = updated_experiments.rename_axis('task_id', axis=0)
    return updated_experiments


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        type=str,
        help="path to the existing file with experiments list",
        default="config/experiments.csv",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="file path to save the experiments list to",
        default=None,
    )

    args = parser.parse_args()

    if args.output_file is None:
        timestamp = datetime.now().strftime("%d_%m_%y__%H_%M_%S")
        if not Path('config').exists():
            Path('config').mkdir()
        output_file = f'config/experiments_{timestamp}.csv'
    else:
        output_file = args.output_file

    df = update_experiments_list(
        input_file=args.input_file,
        experiments=experiments)

    df.to_csv(output_file)

