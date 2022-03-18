
from more_itertools import chunked
import psb2
import os
import wandb

from pbe import program_by_example

DATA_PATH = os.environ['DATA_PATH']
HEAT_UP_RATE = 0.2
MAX_TRIES = 1000
    
with open('tasks.txt') as f:
    task_descriptions = {name.strip(): description.strip() for name, description in chunked(f.readlines(), 2)}

if __name__ == '__main__':
    for problem in psb2.PROBLEMS:
        try:
            wandb.init(project='nl2ml-codex', config={'problem': problem, 'max_tries': MAX_TRIES, 'heat_up_rate': HEAT_UP_RATE})

            description = task_descriptions[problem]
            train_data, test_data = psb2.fetch_examples(DATA_PATH, problem, 5, 2000, format='competitive')
            
            for step, solution, score in program_by_example(problem, description, train_data, test_data, 
                                                            max_options=MAX_TRIES, heat_up_rate=HEAT_UP_RATE):
                wandb.log({'step': step, 'score': score})
                solution.save('solutions/' + problem + '.cpp')
            run.finish()
        except KeyError:
            pass