import logging
import os
import random
from pathlib import Path
from uuid import uuid4

import pandas as pd
import psb2
from fire import Fire
from more_itertools import chunked
from programlib import language_

import wandb
from develop import develop
from github import config_repo, upload_file

logger = logging.getLogger(__name__)

DATA_PATH = os.environ.get('DATA_PATH') or 'psb2'

task_descriptions = []
with open('psb2-meta/tasks.txt') as f:
    task_descriptions = {name.strip(): description.strip()
                         for name, description in chunked(f.readlines(), 2)}

with open('debug-prompt-templates/prompts.txt') as f:
    debug_templates = {int(ix.strip()): prompt.strip() \
                       for ix, prompt in list(map(lambda x: x.split('\t'), f.readlines()))}


def title2kebabcase(title):
    return '-'.join(word.lower() for word in title.split(' '))


pushgp_success_rates = pd.read_csv('psb2-meta/results.tsv',
                                   sep='\t', index_col=['Problem'])
pushgp_success_rates = pushgp_success_rates['Succ.'].rename(title2kebabcase)


def run_benchmark(problem, language='C++', branching_factor=100,
                  max_programs=1000, beam_width=100, debug_prompt_id=0,
                  seed=42, valid_examples=100, test_examples=2000,
                  prompt_examples=5, batch_size=10, mode='execute'):
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
    mode : str
        'execute' or 'debug'

    """
    # Setup logging
    Path('logs').mkdir(exist_ok=True)
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)

    baseline = pushgp_success_rates[problem]
    # Passing locals() as parameter to avoid RecursionError in wandb
    # when config = locals() as a variable, config contains itself recursively
    run = wandb.init(project='nl2ml-codex', config=locals())
    run.config['task_id'] = get_task_id()
    run.config['slurm_job_id'] = os.environ.get('SLURM_JOB_ID')

    language = language_(language)
    os.makedirs('solutions', exist_ok=True)
    solutions_dir = Path('solutions') / str(uuid4())

    # if git env variables are set, this will set up a git repo
    solutions_repo = config_repo(solutions_dir, branch=f'bf{branching_factor}_promptid{debug_prompt_id}')
    # if git env variables are not set, this will just create the directory
    os.makedirs(solutions_dir, exist_ok=True)

    description = task_descriptions[problem]
    debug_prompt_text = debug_templates[debug_prompt_id]

    # ensure that the same I/O pairs are fetched for every experiment
    random.seed(seed)

    train_data, test_data = psb2.fetch_examples(
        DATA_PATH, problem, max(valid_examples, prompt_examples),
        test_examples, format='competitive')
    prompt_data = train_data[:prompt_examples]
    valid_data = train_data[:valid_examples]

    if mode == 'debug':
        for ix in range(5):
            train_data, test_data = psb2.fetch_examples(DATA_PATH, problem, 5, 10, format='competitive')
            for filename, data in zip([f'train_{ix}.txt', f'test_{ix}.txt'], [train_data, test_data]):
                with open(solutions_dir / filename, 'w') as f:
                    f.writelines(list(map(lambda x: '\t'.join([x[0][0], x[1][0]]) + '\n', data)))

    def log_program(solution):
        filename = language.source.format(name=problem)
        solution.save(solutions_dir / filename)
        if solutions_repo:
            idx = wandb.run.summary['idx']
            cmsg = f'solution {idx} of {problem}, {solution.pass_rate} of tests passed'
            upload_file(solutions_repo, filename, cmsg)

    solution = develop(description, prompt_data, valid_data,
                       debug_prompt_text=debug_prompt_text,
                       language=language,
                       beam_width=beam_width,
                       branching_factor=branching_factor,
                       max_programs=max_programs,
                       log_metrics=wandb.log,
                       log_program=log_program,
                       batch_size=min(batch_size, branching_factor))

    solution.test(test_data)
    wandb.log({'test_avg_score': solution.avg_score,
               'test_pass_rate': solution.pass_rate})
    run.finish()


def get_task_id():
    try:
        task_id = os.environ.get('TASK_ID') or int(os.environ.get('SLURM_ARRAY_TASK_ID')) - 1
    except TypeError:
        task_id = None
    return task_id


experiments = [
    {'problem': problem,
     'language': language,
     'branching_factor': branching_factor,
     'max_programs': 1000,
     'beam_width': branching_factor}
    for problem in task_descriptions.keys()
    for language in ('C++', 'Python')
    for branching_factor in (1, 10, 100, 1000)
]

experiments_manual_prompt = [
    {'problem': problem,
     'language': language,
     'branching_factor': branching_factor,
     'max_programs': 1000,
     'beam_width': branching_factor,
     'debug_prompt_id': debug_prompt_id,
     'batch_size': 10}
    for debug_prompt_id in range(11)
    for language in ('C++', 'Python', 'Java')
    for problem in task_descriptions.keys()
    for branching_factor in [1]
]

if __name__ == '__main__':
    task_id = get_task_id()
    logger.info('Start')
    if task_id is not None:
        run_benchmark(**experiments_manual_prompt[int(task_id)])
    else:
        Fire(run_benchmark)
    logger.info('Finish')
