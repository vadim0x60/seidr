import logging
import os
import random
import traceback
import pandas as pd
import psb2
import wandb
from develop import develop
from github import config_repo, upload_file
from fire import Fire
from more_itertools import chunked
from pathlib import Path
from programlib import Program
from programlib import language_
from uuid import uuid4

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

def is_already_solved(solution_path, test_data):
    try:
        with open(solution_path) as f:
            return Program(f.read()).test(test_data) == 1.0
    except FileNotFoundError:
        return False

bf_experiments = [
    {'problem': problem, 
     'language': language, 
     'branching_factor': branching_factor, 
     'max_programs': 1000, 
     'beam_width': branching_factor}
    for problem in task_descriptions.keys()
    for language in ('C++', 'Python')
    for branching_factor in (1, 10, 100, 1000)
]

prompt_experiments = [
    {'problem': problem,
     'language': language,
     'branching_factor': branching_factor,
     'max_programs': 1000,
     'beam_width': branching_factor,
     'debug_prompt_id': debug_prompt_id,
     'batch_size': 10}
    for debug_prompt_id in range(11)
    for language in ('C++', 'Python')
    for problem in task_descriptions.keys()
    for branching_factor in [1]
]

experiments = bf_experiments + prompt_experiments

def run_benchmark(problem='fizz-buzz', language='C++', branching_factor=100,
                  max_programs=1000, beam_width=100, debug_prompt_id=0,
                  seed=42, valid_examples=100, test_examples=2000,
                  prompt_examples=5, batch_size=10, mode='execute', log='ERROR',
                  task_id=None):
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
    
    if task_id:
        return run_benchmark(**{**locals(), **experiments[task_id - 1]})

    # Setup logging
    Path('logs').mkdir(exist_ok=True)
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=log.upper())
    logging.info('logging info')
    baseline = pushgp_success_rates[problem]

    run = wandb.init(entity=os.environ.get('WANDB_ENTITY'), project='codex-for-psb', config=locals())
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

    filename = language.source.format(name=problem)
    if is_already_solved(solutions_dir / filename, test_data):
        logging.info(f'{problem} is already solved, shutting down')
        return

    def log_program(solution):
        solution.save(solutions_dir / filename)
        if solutions_repo:
            idx = wandb.run.summary['idx']
            cmsg = f'solution {idx} of {problem}, {solution.pass_rate} of validation tests passed'
            upload_file(solutions_repo, filename, cmsg)

    call_count = 0
    def log_gpt_call(**kwargs):
        nonlocal call_count
        wandb.log({'gpt_calls': call_count})
        call_count += 1

    solution = develop(description, prompt_data, valid_data,
                       debug_prompt_text=debug_prompt_text,
                       language=language,
                       beam_width=beam_width,
                       branching_factor=branching_factor,
                       max_programs=max_programs,
                       log_metrics=wandb.log,
                       log_program=log_program,
                       log_gpt_call=log_gpt_call,
                       batch_size=min(batch_size, branching_factor))

    logging.info('Development done. Testing...')
    solution.test(test_data)
    wandb.log({'test_avg_score': solution.avg_score,
               'test_pass_rate': solution.pass_rate})
    run.finish()

if __name__ == '__main__':
    slurm_task_id = os.environ.get('SLURM_ARRAY_TASK_ID')
    if slurm_task_id:
        os.environ['TASK_ID'] = slurm_task_id

    logger.info('Start')
    try:
        Fire(run_benchmark)
    except:
        logging.error(traceback.format_exc())
        raise
    logger.info('Finish')