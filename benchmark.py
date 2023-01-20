
from more_itertools import chunked
import psb2
import os
import wandb
import itertools
import pandas as pd
from pathlib import Path
from uuid import uuid4
from programlib import language_, Program

from github import config_repo, upload_file
from develop import develop

from fire import Fire

DATA_PATH = os.environ.get('DATA_PATH') or 'psb2'

with open('psb2-meta/tasks.txt') as f:
    task_descriptions = {name.strip(): description.strip() 
                         for name, description in chunked(f.readlines(), 2)}

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

def run_benchmark(problem, language='C++', branching_factor=100, 
                  max_programs=1000, beam_width=100,
                  prompt_examples=5, valid_examples=100, test_examples=2000):
    baseline = pushgp_success_rates[problem]
    
    config = locals()
    
    language = language_(language)
    os.makedirs('solutions', exist_ok=True)
    solutions_dir = Path('solutions') / str(uuid4())
    # if git env variables are set, this will set up a git repo
    solutions_repo = config_repo(solutions_dir, branch=f'bf{branching_factor}')
    # if git env variables are not set, this will just create the directory
    os.makedirs(solutions_dir, exist_ok=True)

    description = task_descriptions[problem]
    train_data, test_data = psb2.fetch_examples(
        DATA_PATH, problem, max(valid_examples, prompt_examples), 
        test_examples, format='competitive')
    prompt_data = train_data[:prompt_examples]
    valid_data = train_data[:valid_examples]

    filename = language.source.format(name=problem)
    if is_already_solved(solutions_dir / filename, test_data):
        print(f'{problem} is already solved, shutting down')
        return

    run = wandb.init(project='nl2ml-codex', config=config)

    def log_program(solution):
        solution.save(solutions_dir / filename)
        if solutions_repo:
            idx = wandb.run.summary['idx']
            cmsg = f'solution {idx} of {problem}, {solution.pass_rate} of tests passed'
            upload_file(solutions_repo, filename, cmsg)
    
    solution = develop(description, prompt_data, valid_data, 
                       language=language, beam_width=beam_width, 
                       branching_factor=branching_factor, 
                       max_programs=max_programs,
                       log_metrics=wandb.log, log_program=log_program)

    solution.test(test_data)
    wandb.log({'test_avg_score': solution.avg_score,
               'test_pass_rate': solution.pass_rate})

    run.finish()

experiments = [
    {'problem': problem, 
     'language': language, 
     'branching_factor': branching_factor, 
     'max_programs': 1000, 
     'beam_width': branching_factor}
    for problem in task_descriptions.keys()
    for language in ('C++', 'Python', 'Java')
    for branching_factor in (1, 10, 100, 1000)
]

if __name__ == '__main__':
    task_id = os.environ.get('TASK_ID') or os.environ.get('SLURM_ARRAY_TASK_ID')
    if task_id is not None:
        run_benchmark(**experiments[int(task_id)])
    else:
        Fire(run_benchmark)