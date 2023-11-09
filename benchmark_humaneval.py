"""Load and benchmark SEIDR on HumanEval-x"""

import logging
import os
import pathlib
import random
import traceback
from typing import List

import pandas as pd
import psb2
import wandb
from parse_humaneval_tests import load_jsonl
from seidr import get_template
from seidr.dev import develop
from seidr.eval import IOMatch, UnitTest
from seidr.github import FileLogger
from fire import Fire
from more_itertools import chunked
from programlib import Program
from programlib import language_
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_PATH = os.environ.get('DATA_PATH')

debug_templates = [line.split('\t') 
                   for line in get_template('prompts.txt').splitlines()]
debug_templates = {int(ix.strip()): prompt.strip() 
                   for ix, prompt in debug_templates }

def title2kebabcase(title):
    return '-'.join(word.lower() for word in title.split(' '))


pushgp_success_rates = pd.read_csv('psb2-meta/results.tsv',
                                   sep='\t', index_col=['Problem'])
pushgp_success_rates = pushgp_success_rates['Succ.'].rename(title2kebabcase)

def is_already_solved(solutions_logger, test_data, language):
    try:
        return Program(workdir=solutions_logger.dir,
                       name=solutions_logger.filename,
                       language=language).test(test_data)
    except FileNotFoundError:
        return False


def load_humaneval_problem(
        data_path: pathlib.Path,
        language: str = "Python",
        problem: str = "Python/0"
) -> (str, List[str], str):
    """Load prompt, tests and the canonical solution from parsed tests
    from `data_path` path to the `humaneval` folder with jsonl files"""
    language = "cpp" if language.lower() == "c++" else language.lower()
    data = load_jsonl(data_path / f"humaneval_{language}_split_tests.jsonl")
    task = [item for item in data if item["task_id"] == problem][0]
    return task["prompt"], task["tests_split"], task["canonical_solution"]


def run_benchmark(problem='Python/0', language='Python', branching_factor=100,
                  max_programs=1000, beam_width=100, debug_prompt_id=0,
                  seed=42, valid_examples=100, test_examples=2000,
                  prompt_examples=5, batch_size=10, mode='execute', log='ERROR',
                  **kwargs):
    """Generate and repair programs in PSB2

    Parameters
    ----------
    problem : str
        name of a problem in PSB 2
    language : str
        programming language
    branching_factor : int
        number of leaves to create at level n+1
        from each current leaf at level n in the beam
    max_programs : int
        maximum number of elements in the resulting beam
    beam_width : int
        number of elements with top score that will be kept in the beam
        out of all leaves at the newly created level n+1
    debug_prompt_id : int
        prompt template id from `./debug-prompt-templates/prompts.txt
    seed : int
        used to fix seed so that the same I/O pairs are fetched from PSB2
    valid_examples : int
        number of I/O pairs to fetch as train set for program generation,
        upper limit: 1M (set by PSB2), edge cases appear first,
        random pairs are added to fill up the required number
    test_examples : int
        number of I/O pairs to fetch as test set for program generation,
        upper limit: 1M (set by PSB2),
        test cases contain only random I/O pairs, but no edge cases.
    prompt_examples : int
        number of I/O pairs taken from n_train_pairs to generate initial prompt
        for Codex completion model
    batch_size : int
        number of Codex outputs for the same prompt that will be generated at once
        for one parent during the beam search
    mode : str
        'execute' or 'debug'
    """
    # Setup logging
    Path('logs').mkdir(exist_ok=True)
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=log.upper())
    logging.info('logging info')

    config = {
        'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
        'task_id': os.environ.get('TASK_ID'),
        'dataset': f"humaneval-{language}",
        **kwargs,
        **locals()
    }

    del config['kwargs']
    run = wandb.init(entity=os.environ.get('WANDB_ENTITY'), project='codex-for-psb', config=config)
    logger.info(f'Run config {run.config}, W&B: {run.url}')

    language = language_(language)

    commit_msg_template = get_template('commit.txt').format(
        problem=problem,
        wandb_url=run.url)

    attempts_branch = f'human_eval_bf{branching_factor}_promptid{debug_prompt_id}_maxprograms{max_programs}_dev'
    solutions_branch = f'human_eval_bf{branching_factor}_promptid{debug_prompt_id}_maxprograms{max_programs}'

    attempts_logger = FileLogger(branch=attempts_branch, 
                                 filename=language.source.format(name=problem),
                                 commit_msg_template=commit_msg_template)
    solutions_logger = FileLogger(branch=solutions_branch,
                                  filename=language.source.format(name=problem),
                                  commit_msg_template=commit_msg_template)


    debug_template = debug_templates[debug_prompt_id]

    # ensure that the same I/O pairs are fetched for every experiment
    random.seed(seed)

    start_prompt, tests, canonical_solution = load_humaneval_problem(
        pathlib.Path(DATA_PATH), language.name, problem
    )

    prompt_data = tests[:min(prompt_examples, len(tests))]
    valid_data = tests[:min(valid_examples, len(tests))]
    test_data = tests[min(valid_examples, len(tests)):]

    if is_already_solved(solutions_logger, test_data, language):
        logging.info(f'{problem} is already solved, shutting down')
        return

    call_count = 0
    def log_gpt_call(**kwargs):
        nonlocal call_count
        wandb.log({'gpt_calls': call_count})
        call_count += 1

    validation_critics = [
        lambda code: UnitTest(code, language, test) for test in valid_data
    ]

    description = "Complete the following code given the task description and function signature."

    solution = develop(task_description=description,
                       start=start_prompt,
                       critics=validation_critics,
                       language=language,
                       beam_width=beam_width,
                       branching_factor=branching_factor,
                       max_programs=max_programs,
                       log_metrics=wandb.log,
                       log_attempt=attempts_logger,
                       log_solution=solutions_logger,
                       log_gpt_call=log_gpt_call,
                       batch_size=min(batch_size, branching_factor))

    logging.info('Development done. Testing...')
    test_evals = [UnitTest(solution, language, test) for test in valid_data]
    avg_score = sum(e.score() for e in test_evals) / len(test_evals)
    test_pass_rate = sum(e.check() for e in test_evals) / len(test_evals)

    wandb.log({'test_avg_score': avg_score,
               'test_pass_rate': test_pass_rate})
    run.finish()


if __name__ == '__main__':
    try:
        Fire(run_benchmark)
    except:
        logging.error(traceback.format_exc())
        raise