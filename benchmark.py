
from more_itertools import chunked
import psb2
import os
import wandb
import itertools
import pandas as pd
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
                  max_tries=1000, beam_size=100):
    baseline = pushgp_success_rates[problem]
    
    config = locals()
    run = wandb.init(project='nl2ml-codex', config=config)
    
    language = language_(language)
    solutions_repo = ensure_repo(os.environ['GITHUB_REMOTE'], 'solutions', branch=f'bf{branching_factor}')
    solutions_repo.config_writer().set_value('user', 'name', os.environ['GIT_USER']).release()
    solutions_repo.config_writer().set_value('user', 'email', os.environ['GIT_EMAIL']).release()

    description = task_descriptions[problem]
    train_data, test_data = psb2.fetch_examples(DATA_PATH, problem, 5, 2000, format='competitive')
    
    solutionogen = develop(problem, description, train_data, language=language,
                           beam_size=beam_size, branching_factor=branching_factor, 
                           log_f=wandb.log)

    for solution in itertools.islice(solutionogen, max_tries):
        wandb.log({'test_score': solution.test(test_data)})

        filename = language.source.format(name=problem)
        solution.save('solutions/' + filename)
        upload_file(solutions_repo, filename, f'solved {solution.score} of {problem}')
        
    run.finish()

experiments = [
    lambda: run_benchmark(problem, language, branching_factor, 1000, 100)
    for problem in task_descriptions.keys()
    for language in ('C++', 'Python')
    for branching_factor in (1, 100, 1000)
]

if __name__ == '__main__':
    task_id = os.environ.get('TASK_ID') or os.environ.get('SLURM_ARRAY_TASK_ID')
    if task_id is not None:
        experiments[int(task_id)]()
    else:
        Fire(run_benchmark)