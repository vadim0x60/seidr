from urllib import response
import openai
from string import Template
import os
from fire import Fire
from tenacity import retry, retry_if_exception_type, wait_fixed
from programlib import language_

@retry(retry=retry_if_exception_type(openai.error.RateLimitError), wait=wait_fixed(20))
def nl2ml(nl_prompt, language='C++', temperature=0.0):
    """
    Converts a natural language text to C++ programs.
    """
    language = language_(language)

    with open(language.source.format(name='template')) as f:
        template = Template(f.read())

    ml_prompt = template.substitute(prompt=nl_prompt)

    completion = openai.Completion.create(
        engine="code-davinci-001",
        prompt=ml_prompt,
        temperature=temperature,
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

def nl2ml_options(nl_prompt, heat_up_rate=0.2, lang='cpp'):
    return (nl2ml(nl_prompt, temperature=temperature, lang=lang)
            for temperature in heat_up(rate=heat_up_rate))

if __name__ == '__main__':
    Fire(nl2ml)