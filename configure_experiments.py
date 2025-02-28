"""Create a separate CSV file with a list of experiments and their Slurm task id's"""

import argparse

import pandas as pd

from datetime import datetime
from pathlib import Path
from typing import Any, List

from benchmark import task_descriptions

bf_experiments = [
    {
        "problem": problem,
        "language": language,
        "branching_factor": branching_factor,
        "max_programs": 100,
        "beam_width": branching_factor,
        "debug_prompt_id": 0,
        "log": "INFO",
    }
    for branching_factor in (2, 4, 16, 1, 10, 100)
    for problem in task_descriptions.keys()
    for language in ("C++", "Python")
]

humaneval_task_ids = {
    "c++": [f"CPP/{i}" for i in range(164)],
    "python": [f"Python/{i}" for i in range(164)],
}

bf_experiments_humaneval = []

for language in ["Python"]:  # ["C++", "Python"]:
    bf_experiments_humaneval += [
        {
            "problem": problem,
            "language": language,
            "branching_factor": branching_factor,
            "max_programs": 100,
            "beam_width": branching_factor,
            "debug_prompt_id": 0,
            "log": "INFO",
            "dataset": "humaneval",
        }
        for branching_factor in (2, 4, 16, 1, 10, 100)
        for problem in humaneval_task_ids[language.lower()]
    ]

bf_experiments_humaneval_lexicase_py = []

for language in ["Python"]:
    bf_experiments_humaneval_lexicase_py += [
        {
            "problem": problem,
            "language": language,
            "branching_factor": branching_factor,
            "max_programs": 100,
            "beam_width": branching_factor,
            "debug_prompt_id": 0,
            "log": "INFO",
            "lexicase_selection": True,
            "dataset": "humaneval",
        }
        for branching_factor in (2, 4, 16, 10)
        for problem in humaneval_task_ids[language.lower()]
    ]

bf_experiments_humaneval_lexicase_cpp = []

for language in ["C++"]:
    bf_experiments_humaneval_lexicase_cpp += [
        {
            "problem": problem,
            "language": language,
            "branching_factor": branching_factor,
            "max_programs": 100,
            "beam_width": branching_factor,
            "debug_prompt_id": 0,
            "log": "INFO",
            "lexicase_selection": True,
            "dataset": "humaneval",
        }
        for branching_factor in (2, 4, 16, 10)
        for problem in humaneval_task_ids[language.lower()]
    ]

bf_experiments_lexicase = [
    {
        "problem": problem,
        "language": language,
        "branching_factor": branching_factor,
        "max_programs": 100,
        "beam_width": branching_factor,
        "debug_prompt_id": 0,
        "log": "INFO",
        "lexicase_selection": True,
        "dataset": "psb2",
    }
    for branching_factor in (2, 4, 16, 10)
    for problem in task_descriptions.keys()
    for language in ("C++", "Python")
]

bf_psb2_gpt35_no_lexicase = [
    {
        "problem": problem,
        "language": language,
        "branching_factor": branching_factor,
        "max_programs": 100,
        "drafts_per_prompt": branching_factor,
        "explanations_per_program": 2,
        "repairs_per_explanation": branching_factor,
        "beam_width": branching_factor,
        "log": "INFO",
        "lexicase_selection": False,
        "dataset": "psb2",
        "model_name": "gpt-3.5-turbo",
    }
    for branching_factor in (2, 4, 16, 1, 10, 100)
    for problem in task_descriptions.keys()
    for language in ("C++", "Python")
]

bf_psb2_gpt35_lexicase = [
    {
        "problem": problem,
        "language": language,
        "branching_factor": branching_factor,
        "max_programs": 100,
        "drafts_per_prompt": branching_factor,
        "explanations_per_program": 2,
        "repairs_per_explanation": branching_factor,
        "beam_width": branching_factor,
        "log": "INFO",
        "lexicase_selection": True,
        "dataset": "psb2",
        "model_name": "gpt-3.5-turbo",
    }
    for branching_factor in (2, 4, 16, 10)
    for problem in task_descriptions.keys()
    for language in ("C++", "Python")
]

bf_psb2_codellama_no_lexicase = [
    {
        "problem": problem,
        "language": language,
        "branching_factor": branching_factor,
        "max_programs": 100,
        "drafts_per_prompt": branching_factor,
        "explanations_per_program": 2,
        "repairs_per_explanation": branching_factor,
        "beam_width": branching_factor,
        "log": "INFO",
        "lexicase_selection": False,
        "dataset": "psb2",
        "model_name": "codellama:34b-instruct",
    }
    for branching_factor in (2, 4, 16, 1, 10, 100)
    for problem in task_descriptions.keys()
    for language in ("C++", "Python")
]

bf_psb2_codellama_lexicase = [
    {
        "problem": problem,
        "language": language,
        "branching_factor": branching_factor,
        "max_programs": 100,
        "drafts_per_prompt": branching_factor,
        "explanations_per_program": 2,
        "repairs_per_explanation": branching_factor,
        "beam_width": branching_factor,
        "log": "INFO",
        "lexicase_selection": True,
        "dataset": "psb2",
        "model_name": "codellama:34b-instruct",
    }
    for branching_factor in (2, 4, 16, 10)
    for problem in task_descriptions.keys()
    for language in ("C++", "Python")
]

bf_psb2_codellama = bf_psb2_codellama_no_lexicase + bf_psb2_codellama_lexicase


# HumanEval

bf_humaneval_gpt35_no_lexicase_py = []

for language in ["Python"]:
    bf_humaneval_gpt35_no_lexicase_py += [
        {
            "problem": problem,
            "language": language,
            "branching_factor": branching_factor,
            "max_programs": 100,
            "drafts_per_prompt": branching_factor,
            "explanations_per_program": 2,
            "repairs_per_explanation": branching_factor,
            "beam_width": branching_factor,
            "log": "INFO",
            "lexicase_selection": False,
            "dataset": "humaneval",
            "model_name": "gpt-3.5-turbo",
        }
        for branching_factor in (2, 4, 16, 1, 10, 100)
        for problem in humaneval_task_ids[language.lower()]
    ]

bf_humaneval_codellama_no_lexicase_py = []

for language in ["Python"]:
    bf_humaneval_codellama_no_lexicase_py += [
        {
            "problem": problem,
            "language": language,
            "branching_factor": branching_factor,
            "max_programs": 100,
            "drafts_per_prompt": branching_factor,
            "explanations_per_program": 2,
            "repairs_per_explanation": branching_factor,
            "beam_width": branching_factor,
            "log": "INFO",
            "lexicase_selection": False,
            "dataset": "humaneval",
            "model_name": "codellama:34b-instruct",
        }
        for branching_factor in (2, 4, 16, 1, 10, 100)
        for problem in humaneval_task_ids[language.lower()]
    ]


bf_humaneval_gpt35_no_lexicase_cpp = []

for language in ["C++"]:
    bf_humaneval_gpt35_no_lexicase_cpp += [
        {
            "problem": problem,
            "language": language,
            "branching_factor": branching_factor,
            "max_programs": 100,
            "drafts_per_prompt": branching_factor,
            "explanations_per_program": 2,
            "repairs_per_explanation": branching_factor,
            "beam_width": branching_factor,
            "log": "INFO",
            "lexicase_selection": False,
            "dataset": "humaneval",
            "model_name": "gpt-3.5-turbo",
        }
        for branching_factor in (2, 4, 16, 1, 10, 100)
        for problem in humaneval_task_ids[language.lower()]
    ]

bf_humaneval_codellama_no_lexicase_cpp = []

for language in ["C++"]:
    bf_humaneval_codellama_no_lexicase_cpp += [
        {
            "problem": problem,
            "language": language,
            "branching_factor": branching_factor,
            "max_programs": 100,
            "drafts_per_prompt": branching_factor,
            "explanations_per_program": 2,
            "repairs_per_explanation": branching_factor,
            "beam_width": branching_factor,
            "log": "INFO",
            "lexicase_selection": False,
            "dataset": "humaneval",
            "model_name": "codellama:34b-instruct",
        }
        for branching_factor in (2, 4, 16, 1, 10, 100)
        for problem in humaneval_task_ids[language.lower()]
    ]


bf_humaneval_gpt35_lexicase_py = []
for language in ["Python"]:
    bf_humaneval_gpt35_lexicase_py += [
        {
            "problem": problem,
            "language": language,
            "branching_factor": branching_factor,
            "max_programs": 100,
            "drafts_per_prompt": branching_factor,
            "explanations_per_program": 2,
            "repairs_per_explanation": branching_factor,
            "beam_width": branching_factor,
            "log": "INFO",
            "lexicase_selection": True,
            "dataset": "humaneval",
            "model_name": "gpt-3.5-turbo",
        }
        for branching_factor in (2, 4, 16, 10)
        for problem in humaneval_task_ids[language.lower()]
    ]


bf_humaneval_codellama_lexicase_py = []
for language in ["Python"]:
    bf_humaneval_codellama_lexicase_py += [
        {
            "problem": problem,
            "language": language,
            "branching_factor": branching_factor,
            "max_programs": 100,
            "drafts_per_prompt": branching_factor,
            "explanations_per_program": 2,
            "repairs_per_explanation": branching_factor,
            "beam_width": branching_factor,
            "log": "INFO",
            "lexicase_selection": True,
            "dataset": "humaneval",
            "model_name": "codellama:34b-instruct",
        }
        for branching_factor in (2, 4, 16, 10)
        for problem in humaneval_task_ids[language.lower()]
    ]

bf_humaneval_gpt35_lexicase_cpp = []
for language in ["C++"]:
    bf_humaneval_gpt35_lexicase_cpp += [
        {
            "problem": problem,
            "language": language,
            "branching_factor": branching_factor,
            "max_programs": 100,
            "drafts_per_prompt": branching_factor,
            "explanations_per_program": 2,
            "repairs_per_explanation": branching_factor,
            "beam_width": branching_factor,
            "log": "INFO",
            "lexicase_selection": True,
            "dataset": "humaneval",
            "model_name": "gpt-3.5-turbo",
        }
        for branching_factor in (2, 4, 16, 10)
        for problem in humaneval_task_ids[language.lower()]
    ]


bf_humaneval_codellama_lexicase_cpp = []
for language in ["C++"]:
    bf_humaneval_codellama_lexicase_cpp += [
        {
            "problem": problem,
            "language": language,
            "branching_factor": branching_factor,
            "max_programs": 100,
            "drafts_per_prompt": branching_factor,
            "explanations_per_program": 2,
            "repairs_per_explanation": branching_factor,
            "beam_width": branching_factor,
            "log": "INFO",
            "lexicase_selection": True,
            "dataset": "humaneval",
            "model_name": "codellama:34b-instruct",
        }
        for branching_factor in (2, 4, 16, 10)
        for problem in humaneval_task_ids[language.lower()]
    ]


def update_experiments_list(
    input_file: Path | str, experiments: list[dict[str, Any]], offset: int = 0
) -> List[dict[str | Any]]:
    """Append a new set of hyperparameters from `experiments` list
    to the previous experiments taken from `input_file`"""
    new_experiments = pd.DataFrame(experiments)
    try:
        previous_experiments = pd.read_csv(input_file, header=0, index_col=0)
        updated_experiments = pd.concat(
            (previous_experiments, new_experiments), ignore_index=True
        )
    except FileNotFoundError:
        updated_experiments = new_experiments
    updated_experiments.index = list(
        range(1 + offset, updated_experiments.shape[0] + offset + 1)
    )
    updated_experiments = updated_experiments.rename_axis("task_id", axis=0)
    return updated_experiments


if __name__ == "__main__":
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

    offset = 24000
    experiments = bf_humaneval_codellama_lexicase_cpp

    if args.output_file is None:
        timestamp = datetime.now().strftime("%d_%m_%y__%H_%M_%S")
        if not Path("config").exists():
            Path("config").mkdir()
        output_file = f"config/experiments_{timestamp}.csv"
    else:
        output_file = args.output_file

    df = update_experiments_list(
        input_file=args.input_file, experiments=experiments, offset=offset
    )

    df.to_csv(output_file)
