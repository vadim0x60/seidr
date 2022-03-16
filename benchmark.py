from nl2ml import nl2ml, build_prompt

from more_itertools import chunked
import psb2
from programlib import Program
import os

DATA_PATH = os.environ['DATA_PATH']
    
with open('tasks.txt') as f:
    task_descriptions = {name.strip(): description.strip() for name, description in chunked(f.readlines(), 2)}

def test_gpt3_on_psb2(problem, description):
    
    train_data, test_data = psb2.fetch_examples(DATA_PATH, problem, 5, 2000, format='competitive')
    prompt = build_prompt(problem, description, train_data)
    try:
        solution = Program(nl2ml(prompt), language='C++')
        solution.save('solutions/' + problem + '.cpp')
        score = solution.score(test_data)
    except AssertionError:
        score = 0

    return score

if __name__ == '__main__':
    try:
        problem = psb2.PROBLEMS[os.environ['SLURM_ARRAY_TASK_ID']]
        description = task_descriptions[problem]
        score = test_gpt3_on_psb2(problem, description)

        with open('results.txt', 'a') as f:
            # A lock might be required here
            # However, the appended text is small and on UNIX systems the append should be atomic
            # See https://stackoverflow.com/questions/11853551/python-multiple-users-append-to-the-same-file-at-the-same-time
            f.write(problem + ' ' + str(score) + '\n')
    except KeyError:
        pass