
from more_itertools import chunked
import psb2
import os
import wandb
import random
import pandas as pd

from github import ensure_repo, upload_file
from pbe import program_by_example

DATA_PATH = os.environ['DATA_PATH']
HEAT_UP_RATE = 0.2
MAX_TRIES = 1000

with open('psb2-meta/tasks.txt') as f:
    task_descriptions = {name.strip(): description.strip() for name, description in chunked(f.readlines(), 2)}

solutions_repo = ensure_repo(os.environ['GITHUB_REMOTE'], 'solutions')
solutions_repo.config_writer().set_value('user', 'name', os.environ['GIT_USER']).release()
solutions_repo.config_writer().set_value('user', 'email', os.environ['GIT_EMAIL']).release()

def title2kebabcase(title):
    return '-'.join(word.lower() for word in title.split(' '))

pushgp_success_rates = pd.read_csv('psb2-meta/results.tsv', sep='\t', index_col=['Problem'])['Succ.'].rename(title2kebabcase)

if __name__ == '__main__':
    problems = psb2.PROBLEMS.copy()
    random.shuffle(problems)
    for problem in problems:
        try:
            config = {
                'problem': problem, 
                'baseline': pushgp_success_rates[problem],
                'max_tries': MAX_TRIES, 
                'heat_up_rate': HEAT_UP_RATE
                }
            run = wandb.init(project='nl2ml-codex', config=config)

            description = task_descriptions[problem]
            train_data, test_data = psb2.fetch_examples(DATA_PATH, problem, 5, 2000, format='competitive')
            
            for step, solution, score in program_by_example(problem, description, train_data, test_data, 
                                                            max_options=MAX_TRIES, heat_up_rate=HEAT_UP_RATE):
                wandb.log({'step': step, 'score': score})
                solution.save('solutions/' + problem + '.cpp')
                upload_file(solutions_repo, problem + '.cpp', f'solved {score} of {problem}')
                
            run.finish()
        except KeyError:
            pass