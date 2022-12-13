import logging
from string import Template
from programlib import language_

from gpt import explore_gpt, query_gpt


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


def debug_prompt(test_runs, debug_prompt_text, task_description=None):
    mistake = [run for run in test_runs if run.correctness == 0][0]

    if mistake.error_lines:
        return f'Fix {mistake.error_lines}'
    else:
        i = '\\n'.join(mistake.input_lines)
        o = '\\n'.join(mistake.expected_output_lines)
        if 'GPT:' in debug_prompt_text:
            debug_prompt_text = debug_prompt_text.split('GPT:')[1].strip()
            output_lines = '\n'.join([s.decode("utf-8") for s in mistake.output_lines])
            return debug_prompt_text.format(
                s=autocomplete_bug_description(task_description, i, o, output_lines))
        return debug_prompt_text.format(i=i, o=o)


def start_coding(prompt, language='C++'):
    language = language_(language)

    with open('code-templates/' + language.source.format(name='template')) as f:
        template = Template(f.read())

    return template.substitute(prompt=prompt)


def autocomplete_bug_description(task_description, input, expected_output, actual_output):
    """
    Create description of a bug using GPT3 completion.
    """
    code_behaviour = f'The code should solve the following problem: {task_description}\n' \
                     f'The code must return {expected_output} for input {input}, \n' \
                     f'but it returns \n{actual_output} \n' \
                     f'Obviously, the error is that '
    # as a senior software engineer, I give you my feedback
    # TODO: in prompts - # GPT: prompt for gpt3 text -> Fix bugs in code {s}
    bug_description = query_gpt(code='', instruction=None, code_behaviour=code_behaviour, n=1, temperature=0.0)[0]
    logging.info(f'\nBug description - output of query_gpt(...): \n{bug_description}\nBug description finished')

    bug_description_2 = next(explore_gpt(code='', instruction=None, code_behaviour=code_behaviour,
                                         batch_size=1, heat_per_batch=0.0))[0]
    logging.info(f'\nBug description - next(explore_gpt(...)): \n{bug_description_2}\nBug description finished')
    return bug_description
