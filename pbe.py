from programlib import Program
from nl2ml import nl2ml, nl2ml_options

def build_prompt(task, task_description, examples):
    prompt = task
    prompt += '\n\n'
    prompt += task_description
    prompt += '\n\nFor example: \n'
    for sample_inputs, sample_outputs in examples:
        for sample_input in sample_inputs:
            prompt += '> ' + sample_input + '\n'
        for sample_output in sample_outputs:
            prompt += sample_output + '\n'
        prompt += '\n'
    return prompt

def program_by_example(task, task_description, examples, 
                       max_options=1000, heat_up_rate=0.2, lang='cpp'):
    prompt = build_prompt(task, task_description, examples)

    best_score = -1

    for i, option in enumerate(nl2ml_options(prompt, heat_up_rate=heat_up_rate, lang=lang)):
        try:
            program = Program(option, language='C++')
            score = program.score(examples)
        except AssertionError:
            score = 0

        if score > best_score:
            best_score = score
            yield i, program, score

        if score == 1:
            break
        if i > max_options:
            break