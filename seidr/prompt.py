import logging
from pathlib import Path
from string import Template

from programlib import language_

from seidr.gpt import explore_gpt
from seidr import get_template

dont_change = 'Do not change anything'

def initial_prompt(task_description, examples):
    prompt = task_description
    prompt += '\nFor example,'
    for sample_inputs, sample_outputs in examples:
        prompt += ''
        prompt += '\ninput:\n'
        for sample_input in sample_inputs:
            prompt += sample_input + '\n'
        prompt += 'output:'
        for sample_output in sample_outputs:
            prompt += '\n' + sample_output
    return prompt

def gpt_assisted_prompt(debug_prompt_text, task_description, input, expected_output, actual_output):
    """
    Create description of a bug using GPT3 completion.
    """
    assert 'GPT ---' in debug_prompt_text and len(debug_prompt_text.split(' --- ')) >= 3, 'Invalid prompt'
    _, code_behavior, debug_prompt_text = debug_prompt_text.split(' --- ')
    # Form problem description using template
    code_behavior = code_behavior.format(
        t=task_description,
        i=input,
        o=expected_output,
        a=actual_output)

    # Get GPT summary of a bug
    logging.info(f'\nGPT summary of a bug:\n{code_behavior}')
    bug_description = next(explore_gpt(source=code_behavior, instruction=None,
                                       modality='text', batch_size=1, t=0.0, delta_t=0.2))
    # Form debug prompt using template
    debug_prompt_text = debug_prompt_text.format(
        s=bug_description,
        t=task_description,
        i=input,
        o=expected_output,
        a=actual_output)
    return debug_prompt_text


def write_debug_prompt(mistake, debug_prompt_text, task_description=None):
    logging.info('Updating debug prompt')
    output_lines = '\n'.join([s.decode("utf-8") if type(s) == bytes else s for s in mistake.output_lines])

    prompt = ""
    if mistake.correctness < 1:
        if mistake.exit_status:
            error_lines = output_lines
            prompt = f'Fix {error_lines}'

        else:
            i = '\\n'.join(mistake.input_lines)
            o = '\\n'.join(mistake.expected_output_lines)
            if 'GPT ---' in debug_prompt_text:
                
                prompt = gpt_assisted_prompt(
                    debug_prompt_text, task_description, mistake.input_lines,
                    mistake.expected_output_lines, output_lines)
            else:
                prompt = debug_prompt_text.format(i=i, o=o)
    else:
        logging.info('\n\nrun.correctness = 1 for all runs, mistake lines are empty\n\n')
        prompt = dont_change

    logging.info(f'The prompt is: \n{prompt}')
    return prompt


def start_coding(prompt, language='C++'):
    language = language_(language)
    template = language.source.format(name='template')
    template = get_template(template)
    template = Template(template)
    return template.substitute(prompt=prompt)
