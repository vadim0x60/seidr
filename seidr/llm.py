import logging
import os

from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI, ChatOllama
from collections.abc import Iterable
from typing import Callable
import re
from black import format_str, FileMode
from pytest_codeblocks import extract_from_buffer
from io import StringIO

from programlib import Language, language_
from seidr.prompt import create_chat_prompt_template

token_error_message = 'tokens for the input and instruction but the maximum allowed is 3000. ' \
                      'Please reduce the input or instruction length.'

def create_ollama_chain(
        model_name: str = "codellama:7b-instruct",
        temperature: float = 0.0,
        mode: str = "generate"
) -> LLMChain:
    """Create an Ollama chain with a custom prompt"""

    chat_model = ChatOllama(
        model=model_name,
        temperature=temperature
    )

    chat_prompt_template = create_chat_prompt_template(mode)

    return LLMChain(llm=chat_model, prompt=chat_prompt_template)


def extract_codes(
        message_content: str,
        language: Language | str
) -> str:
    """Extract code out of a message and (if Python) format it with black"""

    try:
        code_blocks = list(extract_from_buffer(StringIO(message_content)))
    except RuntimeError as e:
        code_blocks = []

    if not code_blocks:
        yield message_content
    else:
        for code_block in code_blocks:
            if language_(language).name == "Python":
                yield run_black(code_block.code).strip()
            else:
                yield code_block.code.strip()


def run_black(code: str) -> str:
    try:
        return format_str(code, mode=FileMode())
    except Exception as e:
        logging.info(e)
        return code

def create_chain(temperature: float = 0.,
                 mode: str = "generate",
                 model_name: str = "codellama:7b-instruct"):
    chat_prompt_template = create_chat_prompt_template(mode)
    if "gpt" in model_name.lower():
        chat_model = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_organization=os.getenv("OPENAI_ORG")
        )
    elif "llama" in model_name.lower():
        chat_model = ChatOllama(
            model=model_name,
            temperature=temperature
        )

    return LLMChain(llm=chat_model, prompt=chat_prompt_template)


def query_llm(
        language: Language | str,
        temperature: float = 0.,
        mode: str = "generate",
        model_name: str = "codellama:7b-instruct",
        n: int = 1,
        **kwargs
) -> list[str]:
    logging.info(f"Query LLM ({model_name}) in mode {mode} with temperature {temperature}")
    chain = create_chain(temperature=temperature, mode=mode, model_name=model_name)
    kwargs['language'] = language
    result = chain.generate([kwargs for _ in range(n)])

    # Assistants are trained to respond with one message.
    # it is theoretically possible to get more than one message, but it is very unlikely.
    assert all(len(r) == 1 for r in result.generations), "The models are expected to respond with one message"
    result = [r[0].message.content for r in result.generations]

    result_logging = "\n\n".join(result)
    logging.info(f"LLM output: {result_logging}")

    if mode != "explain_bugs":
        result = [c for r in result 
                  for c in extract_codes(message_content=r, language=language)]
        result_logging = "\n\n".join(result)
        logging.info(f"LLM output after code extraction: \n{result_logging}")

    return result


def explore_llm(
        language: Language | str,
        log_llm_call: Callable = lambda **kwargs: None,
        mode: str = "generate",
        model_name: str = "codellama:7b-instruct",
        t: float = 0.0,
        delta_t: float = 0.2,
        batch_size: int = 1,
        **kwargs
) -> Iterable[str]:
    while t <= 1:
        log_llm_call(**locals())
        yield from query_llm(
            language=language,
            temperature=t,
            mode=mode,
            model_name=model_name,
            n=batch_size,
            **kwargs
        )

        t += delta_t

if __name__ == '__main__':
    import itertools
    logging.basicConfig(level=logging.INFO)

    fizz_buzz = """
    Write a Python program that iterates integer numbers
    from 1 to 50. For multiples of three print "Fizz"
    instead of the number and for the multiples of five
    print "Buzz". For numbers which are multiples of both
    three and five print "FizzBuzz".
    """

    for model_name in ['codellama:34b-instruct', 'gpt-3.5-turbo']:
        for code in itertools.islice(explore_llm(language='Python', 
                                                task_name='fizz-buzz', 
                                                task_description=fizz_buzz, 
                                                start_code='', 
                                                model_name=model_name), 10):
            print(code)
