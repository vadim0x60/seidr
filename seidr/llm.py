import logging
import os

from langchain_community.chat_models import ChatOpenAI, ChatOllama
from collections.abc import Iterable
from typing import Callable, Optional
from black import format_str, FileMode
from langchain_core.runnables import RunnableSequence
from pytest_codeblocks import extract_from_buffer
from io import StringIO

from programlib import Language, language_
from seidr.prompt import create_chat_prompt_template, ollama_messages

token_error_message = 'tokens for the input and instruction but the maximum allowed is 3000. ' \
                      'Please reduce the input or instruction length.'


def extract_codes(
        message_content: str,
        language: Language | str
) -> str:
    """Extract code out of a message and (if Python) format it with black"""
    try:
        code_blocks = list(extract_from_buffer(StringIO(message_content)))
        code_blocks = [code for code in code_blocks if bool(code)]
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
    """Format (Python) code with Black"""
    try:
        return format_str(code, mode=FileMode())
    except Exception as e:
        logging.info(e)
        return code


def create_chain(
        temperature: float = 0.,
        mode: str = "generate",
        model_name: str = "codellama:7b-instruct",
        base_url: Optional[str] = None
) -> RunnableSequence:
    """Set up a LangChain LLMChain"""
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
            base_url=base_url,
            model=model_name,
            temperature=temperature
        )

    return chat_prompt_template | chat_model


def query_llm(
        language: Language | str,
        base_url: str,
        temperature: float = 0.,
        mode: str = "generate",
        model_name: str = "codellama:7b-instruct",
        n: int = 1,
        **kwargs
) -> list[str]:
    """Generate `n` outputs with an LLM"""
    logging.info(f"Query LLM ({model_name}) in mode {mode} with temperature {temperature}\n")

    kwargs['language'] = str(language)

    # if "gpt" in model_name.lower():
    chain = create_chain(temperature=temperature, mode=mode, model_name=model_name, base_url=base_url)
    result = chain.batch([kwargs for _ in range(n)])

    # Assistants are trained to respond with one message.
    # it is theoretically possible to get more than one message, but it is very unlikely.
    result = [r.content for r in result]

    # elif "llama" in model_name.lower():
    #     messages = ollama_messages(mode, **kwargs)
    #     responses = [ollama.chat(model=model_name, messages=messages, options={"temperature": temperature}) for _ in range(n)]
    #     result = [r['message']['content'] for r in responses]

    if mode == "repair":
        logging.info(f"Generating repair candidates for bug summary: \n{kwargs['bug_summary']}\n")
    elif mode == "explain_bugs":
        logging.info(f"Generating explanations for code: \n{kwargs['code']}\n")

    if mode != "explain_bugs":
        result = [c for r in result
                  for c in extract_codes(message_content=r, language=language)]
        result_logging = "\n\n".join(result)
        logging.info(f"LLM output after code extraction: \n{result_logging}\n")
    else:
        result_logging = "\n\n".join(result)
        logging.info(f"LLM output: \n{result_logging}\n")

    return result


def explore_llm(
        language: Language | str,
        base_url: str,
        log_llm_call: Callable = lambda **kwargs: None,
        mode: str = "generate",
        model_name: str = "codellama:7b-instruct",
        t: float = 0.0,
        delta_t: float = 0.2,
        batch_size: int = 1,
        **kwargs
) -> Iterable[str]:
    """Generate LLM outputs and increase temperature for every new batch"""
    while t <= 1:
        log_llm_call(**locals())
        yield from query_llm(
            language=language,
            temperature=t,
            mode=mode,
            model_name=model_name,
            n=batch_size,
            base_url=base_url,
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
