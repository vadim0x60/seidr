from dis import Instruction
from urllib.request import urlopen
from programlib import Program
import psb2
import os
import openai

def debug_program(source, test_data, language='C++'):
    program = Program(source, name='luhn', language=language)
    score = program.score(test_data)
    mistake = [run for run in program.test_runs if run.correctness == 0][0]

    i = '\\n'.join(mistake.input_lines)
    o = '\\n'.join(mistake.expected_output_lines)

    new_source = openai.Edit.create(
        engine="code-davinci-edit-001",
        input=source,
        instruction=f'Make sure that {i} -> {o}',
        temperature=0.0
        )["choices"][0]["text"]

    return new_source

def debug_program_from_repo(task):
    with urlopen(f'https://raw.githubusercontent.com/psb2/psb2/master/programs/{task}.cpp') as f:
        source = f.read().decode('utf-8')
        
    train_data, test_data = psb2.fetch_examples(
        os.environ['DATA_PATH'], 'bowling', 0, 100, format='competitive')
    new_source = debug_program(source, test_data)
    new_program = Program(new_source, name='luhn2', language='C++')
    print(new_source)
    print(f'Score {new_program.score(test_data)}')

if __name__ == '__main__':
    debug_program_from_repo('bowling')