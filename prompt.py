from string import Template
from programlib import language_
from pathlib import Path
from gpt import explore_gpt

def initial_prompt(task_description, examples, language=None):
    prompt = task_description
    prompt += '\n\nFor example: \n'

    if language:
        prompt += '\nWrite the solution in ' + language

    for sample_inputs, sample_outputs in examples:
        for sample_input in sample_inputs:
            prompt += '> ' + sample_input + '\n'
        for sample_output in sample_outputs:
            prompt += sample_output + '\n'
        prompt += '\n'
    return prompt


def gpt_assisted_prompt(debug_prompt_text, task_description, input, expected_output, actual_output):
    """
    Create description of a bug using GPT3 completion.
    """
    assert 'GPT ---' in debug_prompt_text and len(debug_prompt_text.split(' --- ') >= 3), 'Invalid prompt'
    _, code_behavior, debug_prompt_text = debug_prompt_text.split(' --- ')
    # Form problem description using template
    code_behavior = code_behavior.format(
        t=task_description,
        i=input,
        o=expected_output,
        a=actual_output)

    # Get GPT summary of a bug
    # bug_description = query_gpt(code='', code_behaviour=code_behaviour, n=1, temperature=0.0)[0]
    bug_description = next(explore_gpt(source=code_behavior, modality='text', batch_size=1, heat_per_batch=0.0))

    # Form debug prompt using template
    debug_prompt_text = debug_prompt_text.format(
        s=bug_description,
        t=task_description,
        i=input,
        o=expected_output,
        a=actual_output)
    return debug_prompt_text


def write_debug_prompt(test_runs, debug_prompt_text, task_description=None):
    mistake = min(test_runs, key=lambda run: run.correctness)

    if mistake.error_lines:
        return f'Fix {mistake.error_lines}'
    else:
        i = '\\n'.join(mistake.input_lines)
        o = '\\n'.join(mistake.expected_output_lines)
        if 'GPT ---' in debug_prompt_text:
            output_lines = '\n'.join([s.decode("utf-8") for s in mistake.output_lines])
            return gpt_assisted_prompt(
                debug_prompt_text, task_description, mistake.input_lines, 
                mistake.expected_output_lines, output_lines)
        return debug_prompt_text.format(i=i, o=o)

def start_coding(prompt, language='C++', temperature=0.0):
    language = language_(language)
    template_name = language.source.format(name='template')

    with open(Path('code-templates') / template_name) as f:
        template = Template(f.read())

    return template.substitute(prompt=prompt)