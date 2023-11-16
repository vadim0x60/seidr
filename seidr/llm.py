import logging
import os

import langchain.adapters.openai
import openai
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI, ChatOllama
from langchain.schema import HumanMessage, SystemMessage
from tenacity import retry, retry_if_exception_type, stop_after_attempt
from tenacity import wait_random_exponential, before_sleep_log
import traceback
from collections.abc import Iterable
from typing import Callable
import re

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


def postprocess_code(
        code: str,
        language: str,
        mode: str = "full"
) -> str:
    """Remove quotes or language name around the generated code"""
    code = code.strip()

    if "```" in code:
        matches = [m.group(1) for m in re.finditer("```" + language + "([\w\W]*?)```", code)]
        if len(matches) > 0:
            code = matches[0]
        elif code.startswith("```"):
            matches = [m.group(1) for m in re.finditer("```([\w\W]*?)```", code)]
            if len(matches) > 0:
                code = matches[0]
            else:
                code = code[3:]

    if code.endswith("```") or code.endswith('"""'):
        code = code[:-3]

    if language.lower() == "python":
        code = run_black(code)

    return code.strip()


def run_black(code: str, language: str) -> str:
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
        language: str,
        temperature: float = 0.,
        mode: str = "generate",
        model_name: str = "codellama:7b-instruct",
        n: int = 1,
        **kwargs
) -> list[str]:
    chain = create_chain(temperature=temperature, mode=mode, model_name=model_name)
    kwargs['language'] = language
    result = chain.generate([kwargs for _ in range(n)])
    assert all(len(r) == 1 for r in result.generations), "The models are expected to respond with one message"
    result = [r[0].message.content for r in result.generations]

    #if mode != "explain_bugs":
    #    result = postprocess_code(code=result, language=language)

    return result


def explore_llm(
        language: str,
        log_llm_call: Callable = lambda **kwargs: None,
        mode: str = "generate",
        model_name: str = "codellama:7b-instruct",
        t: float = 0.0,
        delta_t: float = 0.2,
        batch_size: int = None,
        **kwargs
) -> Iterable[str]:
    if not batch_size:
        if 'gpt' in model_name:
            batch_size = 10
        else:
            # Because Ollama doesn't support batch inference
            batch_size = 1

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

    for code in itertools.islice(explore_llm(language='Python', 
                                             problem_name='hello-world', 
                                             task_description='Write a Python program that outputs Hello World', 
                                             start_code='', 
                                             model_name='gpt-3.5-turbo'), 10):
        print(code)
