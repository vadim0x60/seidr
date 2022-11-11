from string import Template
from programlib import language_

def initial_prompt(task, task_description, examples):
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

def debug_prompt(program):
    if program.stderr:
        return f'Fix {program.stderr}'
    else:
        mistake = [run for run in program.test_runs if run.correctness == 0][0]

        i = '\\n'.join(mistake.input_lines)
        o = '\\n'.join(mistake.expected_output_lines)

        return f'Make sure that {i} -> {o}'

def start_coding(prompt, language='C++', temperature=0.0):
    language = language_(language)

    with open(language.source.format(name='template')) as f:
        template = Template(f.read())

    return template.substitute(prompt=prompt)