"""This is a script to generate csv tables with hyperparameters for SEIDR experiments that
span several LLMs,
two datasets,
two programming languages (Python and C++),
 six branching factors and two ranking strategies - all to rerun 10 times"""
import random
from pathlib import Path
from typing import List, Any

import pandas as pd

from benchmark import task_descriptions
from configure_experiments import humaneval_task_ids

branching_factor = {
    "lexicase": {
        "psb2":
            {
                "gpt-3.5-turbo":
                    {
                        "C++": [2],
                        "Python": [16]
                    },
                "llama3":
                    {
                        "C++": [16],
                        "Python": [16] # 100
                    }
            },
        "humaneval":
            {
                "gpt-3.5-turbo":
                    {
                        "C++": [2],
                        "Python": [4]
                    },
                "llama3":
                    {
                        "C++": [10],
                        "Python": [10]
                    }
            }
    },
    "no_lexicase": {
        "psb2":
            {
                "gpt-3.5-turbo":
                    {
                        "C++": [2, 4, 16, 1, 10, 100],
                        "Python": [2, 4, 16, 1, 10, 100]
                    },
                "llama3":
                    {
                        "C++": [2, 4, 16, 1, 10, 100],
                        "Python": [2, 4, 16, 1, 10, 100]
                    }
            },
        "humaneval":
            {
                "gpt-3.5-turbo":
                    {
                        "C++": [2, 4, 16, 1, 10, 100],
                        "Python": [2, 4, 16, 1, 10, 100]
                    },
                "llama3":
                    {
                        "C++": [2, 4, 16, 1, 10, 100],
                        "Python": [2, 4, 16, 1, 10, 100]
                    }
            }
        }
    }

dataset_id = {
    "psb2": 1,
    "humaneval": {
        "lexicase": {
            "Python": 2,
            "C++": 3
        },
        "no_lexicase": {
            "Python": 4,
            "C++": 5
        }
    }
}

model_id = {
    "gpt-3.5-turbo": 1,
    "codellama:34b-instruct": 2,
    "llama3": 3,
    "gpt-4o-2024-05-13": 4
}

queue_names = [
    # "defq",
    # "milanq",
    # "rome16q",
    "mi50q",
    "mi100q",
    "mi210q",
    "slowq"
]

number_of_experiments = {
    "psb2": {
        "no_lexicase": 25 * 2 * len(branching_factor["lexicase"]),
        "lexicase": 25 * 2
    },
    "humaneval": {
        "no_lexicase": 164 * len(branching_factor["lexicase"]), # only one language at a time
        "lexicase": 164
    }
}

def update_experiments_list(
        input_file: Path | str,
        experiments: list[dict[str, Any]],
        offset: int = 0
) -> List[dict[str | Any]]:
    """Append a new set of hyperparameters from `experiments` list
     to the previous experiments taken from `input_file`"""
    new_experiments = pd.DataFrame(experiments)
    try:
        previous_experiments = pd.read_csv(input_file, header=0, index_col=0)
        updated_experiments = pd.concat((previous_experiments, new_experiments), ignore_index=True)
    except FileNotFoundError:
        updated_experiments = new_experiments
    updated_experiments.index = list(range(1 + offset, updated_experiments.shape[0] + offset + 1))
    updated_experiments = updated_experiments.rename_axis('task_id', axis=0)
    return updated_experiments


def update_sbatch_file(
        model_name: str,
        output_file: Path | str,
        dataset: str,
        lexicase_tag: str,
        run: int,
        offset: int,
        language: str = None
) -> None:
    """Create a new sbatch file with a random CPU queue name, and an offset corresponding to the type of experiment"""
    sbatch_template_file = f"scripts/tests/run_1_psb2_llama3_8b.sbatch" if "llama" in model_name \
        else f"scripts/tests/run_1_psb2_gpt4o.sbatch"

    with open(sbatch_template_file, "r") as f:
        sbatch_lines = f.readlines()
        queue = queue_names[random.randint(0, len(queue_names) - 1)]
        sbatch_lines[
            2] = f"#SBATCH --partition={queue}          # partition (queue, see info about the queues below)\n"
        sbatch_lines[8] = f"#SBATCH --array=1-{number_of_experiments[dataset][lexicase_tag]}%5\n"
        sbatch_lines[36] = f'CONFIG="{output_file[2:]}"\n'
        sbatch_lines[37] = f"OFFSET={offset}\n"
        if dataset == "humaneval":
            sbatch_lines[57] = "\n"
            sbatch_lines[75] = ""
            sbatch_lines[95] = ""
            sbatch_lines[61] = 'echo "srun python3 benchmark_humaneval.py \\\n'
            sbatch_lines[81] = 'srun python3 benchmark_humaneval.py \\\n'
            sbatch_script = Path(
                f"scripts/{model_name}/{dataset}/run_{run}_{dataset}_{model_name}_{lexicase_tag}_{language}_{queue}.sbatch")
        else:
            sbatch_script = Path(f"scripts/{model_name}/{dataset}/run_{run}_{dataset}_{model_name}_{lexicase_tag}_{queue}.sbatch")
    sbatch_script.parent.mkdir(parents=True, exist_ok=True)
    with open(sbatch_script, "w") as f:
        f.writelines(sbatch_lines)


if __name__ == '__main__':

    random.seed(42)
    runs = 6

    # lexicase_selection = False
    lexicase_selection = True

    dataset = "psb2"

    for run in range(1, runs + 1):
        for model_name, model_name_folder in zip(
                ["gpt-3.5-turbo", "llama3"],
                ["gpt3_5", "llama3_8b"]
        ):

            lexicase_tag = "lexicase" if lexicase_selection else "no_lexicase"
            offset = int(1000 * (run * 100 + model_id[model_name] * 10 + dataset_id[dataset])) * \
                     (1 + 9 * int(lexicase_selection))

            # bf_psb2_no_lexicase = [
            #     {
            #         'problem': problem,
            #         'language': language,
            #         'branching_factor': bf,
            #         'max_programs': 100,
            #         'drafts_per_prompt': bf,
            #         'explanations_per_program': 2,
            #         'repairs_per_explanation': bf,
            #         'beam_width': bf,
            #         'log': 'INFO',
            #         'lexicase_selection': lexicase_selection,
            #         'dataset': dataset,
            #         'model_name': model_name,
            #         'run': run,
            #         'valid_examples': 50,
            #         'offset': offset
            #     }
            #     for bf in (2, 4, 16, 10)
            #     for problem in task_descriptions.keys()
            #     for language in ('C++', 'Python')
            # ]
            #
            # experiments = bf_psb2_no_lexicase

            bf_psb2_lexicase = [
                {
                    'problem': problem,
                    'language': language,
                    'branching_factor': bf,
                    'max_programs': 100,
                    'drafts_per_prompt': bf,
                    'explanations_per_program': 2,
                    'repairs_per_explanation': bf,
                    'beam_width': bf,
                    'log': 'INFO',
                    'lexicase_selection': True,
                    'dataset': dataset,
                    'model_name': model_name,
                    'run': run,
                    'valid_examples': 50,
                    'offset': offset
                }
                for language in ('C++', 'Python')
                for bf in branching_factor[lexicase_tag][dataset][model_name][language]
                for problem in task_descriptions.keys()

            ]

            experiments = bf_psb2_lexicase

            df = update_experiments_list(
                input_file="config/experiments.csv",
                experiments=experiments,
                offset=offset
            )

            # NO lexicase
            # output_file = (f"./config/{dataset}/{model_name_folder}/experiments_"
            #                f"{dataset}_{model_name_folder}_"
            #                f"run_{run}_offset_{offset}_{lexicase_tag}_"
            #                f"mp100_bf_2_4_16_1_10_100.csv")

            # lexicase
            output_file = (f"./config/{dataset}/{model_name_folder}/experiments_"
                           f"{dataset}_{model_name_folder}_"
                           f"run_{run}_offset_{offset}_{lexicase_tag}_"
                           f"mp100_bf_custom.csv")

            Path(output_file).parent.mkdir(parents=True, exist_ok=True)

            df.to_csv(output_file)

            update_sbatch_file(model_name, output_file, dataset, lexicase_tag, run, offset)


    dataset = "humaneval"

    for run in range(1, runs + 1):
        for model_name, model_name_folder in zip(
                ["gpt-3.5-turbo", "llama3"],
                ["gpt3_5", "llama3_8b"]
        ):

            lexicase_tag = "lexicase" if lexicase_selection else "no_lexicase"

            for language, language_tag in zip(
                    ["C++", "Python"], ["cpp", "py"]
            ):
                offset = int(1000 * (
                        run * 100 + model_id[model_name] * 10 + dataset_id[dataset][lexicase_tag][language])) * \
                        (1 + 9 * int(lexicase_selection))

                experiments = [
                    {
                        'problem': problem,
                        'language': language,
                        'branching_factor': bf,
                        'max_programs': 100,
                        'drafts_per_prompt': bf,
                        'explanations_per_program': 2,
                        'repairs_per_explanation': bf,
                        'beam_width': bf,
                        'log': 'INFO',
                        'lexicase_selection': lexicase_selection,
                        'dataset': dataset,
                        'model_name': model_name,
                        'run': run,
                        'offset': offset
                    }
                    for bf in branching_factor[lexicase_tag][dataset][model_name][language]
                    for problem in humaneval_task_ids[language.lower()]
                ]

                df = update_experiments_list(
                    input_file="config/experiments.csv",
                    experiments=experiments,
                    offset=offset
                )
                output_file = (f"./config/{dataset}/{model_name_folder}/experiments_"
                               f"{dataset}_{model_name_folder}_"
                               f"run_{run}_offset_{offset}_"
                               f"{lexicase_tag}_{language_tag}_"
                               f"mp100_bf_"+"_".join([str(s) for s in branching_factor[lexicase_tag][dataset][model_name][language]])+".csv")

                Path(output_file).parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(output_file)
                update_sbatch_file(model_name, output_file, dataset, lexicase_tag, run, offset, language_tag)

