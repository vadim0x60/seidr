
from more_itertools import chunked
import psb2
import os
import wandb
import itertools
import pandas as pd
from pathlib import Path
from uuid import uuid4
from programlib import language_

from github import ensure_repo, upload_file
from develop import develop

from fire import Fire

DATA_PATH = os.environ.get('DATA_PATH') or 'psb2'

with open('psb2-meta/tasks.txt') as f:
    task_descriptions = {name.strip(): description.strip() for name, description in chunked(f.readlines(), 2)}

def title2kebabcase(title):
    return '-'.join(word.lower() for word in title.split(' '))

pushgp_success_rates = pd.read_csv('psb2-meta/results.tsv', sep='\t', index_col=['Problem'])['Succ.'].rename(title2kebabcase)

def run_benchmark(problem, language='C++', branching_factor=100, 
                  max_tries=1000, beam_size=100,
                  prompt_examples=5, valid_examples=100, test_examples=2000):
    baseline = pushgp_success_rates[problem]
    
    config = locals()
    run = wandb.init(project='nl2ml-codex', config=config)
    
    language = language_(language)
    os.makedirs('solutions', exist_ok=True)
    solutions_dir = Path('solutions') / str(uuid4())
    solutions_repo = ensure_repo(os.environ['GITHUB_REMOTE'], solutions_dir, branch=f'bf{branching_factor}')
    solutions_repo.config_writer().set_value('user', 'name', os.environ['GIT_USER']).release()
    solutions_repo.config_writer().set_value('user', 'email', os.environ['GIT_EMAIL']).release()

    description = task_descriptions[problem]
    train_data, test_data = psb2.fetch_examples(DATA_PATH, problem, max(valid_examples, prompt_examples), test_examples, format='competitive')
    prompt_data = train_data[:prompt_examples]
    valid_data = train_data[:valid_examples]
    
    solutionogen = develop(description, prompt_data, valid_data, 
                           language=language, beam_size=beam_size, 
                           branching_factor=branching_factor, 
                           log_f=wandb.log)

    for idx, solution in enumerate(solutionogen):
        solution.test(test_data)
        wandb.log({'test_avg_score': solution.avg_score,
                   'test_pass_rate': solution.pass_rate})

        filename = language.source.format(name=problem)
        solution.save(solutions_dir / filename)
        upload_file(solutions_repo, filename, f'solution {idx} of {problem}, {solution.pass_rate} of tests passed')

        if idx >= max_tries:
            break
        
    run.finish()

experiments = [
    {'problem': problem, 
     'language': language, 
     'branching_factor': branching_factor, 
     'max_tries': 1000, 
     'beam_size': branching_factor}
    for problem in task_descriptions.keys()
    for language in ('C++', 'Python')
    for branching_factor in (1, 10, 100, 1000)
]

if __name__ == '__main__':
    task_id = os.environ.get('TASK_ID') or os.environ.get('SLURM_ARRAY_TASK_ID')
    if task_id is not None:
        run_benchmark(**experiments[int(task_id)])
    else:
        Fire(run_benchmark)