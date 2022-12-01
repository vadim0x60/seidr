from string import Template
from programlib import language_

def initial_prompt(task, task_description, examples, n_pairs_in_prompt):
    prompt = task
    prompt += '\n\n'
    prompt += task_description
    prompt += '\n\nFor example: \n'
    for sample_inputs, sample_outputs in examples[:n_pairs_in_prompt]:
        for sample_input in sample_inputs:
            prompt += '> ' + sample_input + '\n'
        for sample_output in sample_outputs:
            prompt += sample_output + '\n'
        prompt += '\n'
    return prompt

def debug_prompt(test_runs, debug_prompt_text):
    mistake = [run for run in test_runs if run.correctness == 0][0]

    if mistake.error_lines:
        return f'Fix {mistake.error_lines}'
    else:
        i = '\\n'.join(mistake.input_lines)
        o = '\\n'.join(mistake.expected_output_lines)

        return debug_prompt_text.format(i=i, o=o)

def start_coding(prompt, language='C++', temperature=0.0):
    language = language_(language)

    with open('code-templates/' + language.source.format(name='template')) as f:
        template = Template(f.read())

    return template.substitute(prompt=prompt)
