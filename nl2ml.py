import openai
from string import Template
import os

openai.api_key = os.getenv("OPENAI")

with open('template.cpp') as f:
    template = Template(f.read())

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

def nl2ml(nl_text):
    """
    Converts a natural language text to C++ programs.
    """

    prompt = template.substitute(prompt=nl_text)

    completion = openai.Completion.create(
        engine="code-davinci-001",
        prompt=prompt,
        temperature=0,
        max_tokens= 2000,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=["\"\"\""]
        )["choices"][0]["text"]
    return prompt + '\n' + completion

if __name__ == '__main__':
    print(nl2ml(input()))