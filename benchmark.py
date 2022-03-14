from nl2ml import nl2ml

from more_itertools import chunked
import psb2
from programlib import Program
import os

DATA_PATH = os.environ['DATA_PATH']

def build_prompt(task, task_description, examples):
    prompt = task
    prompt += '\n\n'
    prompt += task_description
    prompt += '\n\nFox example: \n'
    for sample_inputs, sample_outputs in examples:
        for sample_input in sample_inputs:
            prompt += '> ' + sample_input + '\n'
        for sample_output in sample_outputs:
            prompt += sample_output + '\n'
        prompt += '\n'
    return prompt
    
with open('tasks.txt') as f:
    task_descriptions = {name.strip(): description.strip() for name, description in chunked(f.readlines(), 2)}

if __name__ == '__main__':
    scores = {}

    for problem in psb2.PROBLEMS:
        train_data, test_data = psb2.fetch_examples(DATA_PATH, problem, 0, 2000, format='competitive')
        header_txt = problem.replace('-', ' ') + '\n\n' + task_descriptions[problem]
        solution = Program(nl2ml(header_txt), language='C++')
        solution.save('solutions/' + problem + '.cpp')
        score = solution.score(test_data)

        with open('results.txt', 'a') as f:
            f.write(problem + ' ' + str(score) + '\n')