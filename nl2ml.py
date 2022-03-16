import openai
from string import Template
import os

openai.api_key = os.getenv("OPENAI")

with open('template.cpp') as f:
    template = Template(f.read())

def nl2ml(nl_prompt, temperature=0):
    """
    Converts a natural language text to C++ programs.
    """

    ml_prompt = template.substitute(prompt=nl_prompt)

    completion = openai.Completion.create(
        engine="code-davinci-001",
        prompt=ml_prompt,
        temperature=0,
        max_tokens= 2000,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=["\"\"\""]
        )["choices"][0]["text"]
    return ml_prompt + '\n' + completion

def heat_up(rate=0.2):
    temperature = 0

    while True:
        temperature = temperature + (1 - temperature) * rate
        yield temperature

def nl2ml_options(nl_prompt, heat_up_rate=0.2):
    return (nl2ml(nl_prompt, temperature=temperature)
            for temperature in heat_up(rate=heat_up_rate))

if __name__ == '__main__':
    print(nl2ml(input()))