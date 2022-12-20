import logging
from string import Template

from programlib import language_

from gpt import explore_gpt


def initial_prompt(task_description, examples):
    prompt = task_description
    prompt += '\n\nFor example: \n'
    for sample_inputs, sample_outputs in examples:
        for sample_input in sample_inputs:
            prompt += '> ' + sample_input + '\n'
        for sample_output in sample_outputs:
            prompt += sample_output + '\n'
        prompt += '\n'
    return prompt


def gpt_assisted_prompt(debug_prompt_text, task_description, input_line, expected_output, actual_output):
    """
    Create description of a bug using GPT3 completion.
    """
    assert 'GPT ---' in debug_prompt_text and len(debug_prompt_text.split(' --- ')) >= 3, 'Invalid prompt'
    _, code_behaviour, debug_prompt_text = debug_prompt_text.split(' --- ')
    # Form problem description using template
    code_behaviour = code_behaviour.format(
        t=task_description,
        i=input_line,
        o=expected_output,
        a=actual_output)

    # Get GPT summary of a bug
    # bug_description = query_gpt(code='', code_behaviour=code_behaviour, n=1, temperature=0.0)[0]
    bug_description = next(explore_gpt(code='', code_behaviour=code_behaviour, batch_size=1, heat_per_batch=0.0))

    # Form debug prompt using template
    debug_prompt_text = debug_prompt_text.format(
        s=bug_description,
        t=task_description,
        i=input_line,
        o=expected_output,
        a=actual_output)
    return debug_prompt_text


def debug_prompt(test_runs, debug_prompt_text, task_description=None):

    logging.info('Updating debug prompt')
    mistake = [run for run in test_runs if run.correctness == 0][0]
    if mistake.error_lines:
        return f'Fix {mistake.error_lines}'
    else:
        i = '\\n'.join(mistake.input_lines)
        o = '\\n'.join(mistake.expected_output_lines)
        if 'GPT ---' in debug_prompt_text:
            output_lines = '\n'.join([s.decode("utf-8") for s in mistake.output_lines])
            return gpt_assisted_prompt(
                debug_prompt_text, task_description, mistake.input_lines, mistake.expected_output_lines, output_lines)
        return debug_prompt_text.format(i=i, o=o)


def start_coding(prompt, language='C++'):
    language = language_(language)

    with open('code-templates/' + language.source.format(name='template')) as f:
        template = Template(f.read())

    return template.substitute(prompt=prompt)