import logging
import os

import langchain.adapters.openai
import openai
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI, ChatOllama
from tenacity import retry, retry_if_exception_type, stop_after_attempt
from tenacity import wait_random_exponential, before_sleep_log
import traceback
from typing import Generator, Callable
import re

from prompt import create_chat_prompt_template

token_error_message = 'tokens for the input and instruction but the maximum allowed is 3000. ' \
                      'Please reduce the input or instruction length.'


@retry(retry=retry_if_exception_type(openai.error.APIError) |
             retry_if_exception_type(openai.error.APIConnectionError) |
             retry_if_exception_type(openai.error.ServiceUnavailableError),
       wait=wait_random_exponential(max=300),
       stop=stop_after_attempt(5),
       before_sleep=before_sleep_log(logging.getLogger(), logging.ERROR))
@retry(retry=retry_if_exception_type(openai.error.RateLimitError),
       wait=wait_random_exponential(max=600),
       # We want the experiment to crash in this situation
       # We've launched too many
       stop=stop_after_attempt(5),
       before_sleep=before_sleep_log(logging.getLogger(), logging.INFO))
def query_gpt(source=None, instruction=None, modality='code', n=1, t=1.0):
    """
    Get code snippets from GPT-3.

    If instruction is not specified, the code is extended (autocompleted),
    otherwise it's edited according to the instruction.
    """
    logging.info(f"Querying GPT with temperature {t} and {n} snippets.")

    result = []
    try:
        if instruction:
            if modality == 'text':
                engine = "text-davinci-edit-001"
                logging.info("Calling GPT for text editing")

            elif modality == 'code':
                engine = "code-davinci-edit-001"
                logging.info("Calling GPT for code editing")
            else:
                raise ValueError(f'Unknown modality: {modality}')

            response = openai.Edit.create(
                engine=engine,
                input=source,
                n=n,
                instruction=instruction,
                temperature=t
            )
            result = [choice['text'] for choice in response["choices"]
                      if "text" in choice.keys()]
        else:
            if modality == 'text':
                engine = "text-davinci-003"
                logging.info("Calling GPT for text completion")
            elif modality == 'code':
                engine = "code-davinci-002"
                logging.info("Calling GPT for code completion")
            else:
                raise ValueError(f'Unknown modality: {modality}')

            response = openai.Completion.create(
                engine=engine,
                prompt=source,
                n=n,
                temperature=t
            )

            if modality == 'text':
                result = [choice['text'] for choice in response["choices"]
                          if "text" in choice.keys()]
                logging.info(f"\nBug summary by GPT:\n{result[0]}")
            else:
                result = [source + '\n' + choice['text'] for choice in response["choices"]
                          if "text" in choice.keys()]

    except openai.error.InvalidRequestError as e:
        result = []
        if token_error_message in e.error.message:
            raise e

    return result

def explore_gpt(source='', instruction=None, modality='code', batch_size=1,
                t=0.0, delta_t=0.2, log_gpt_call=lambda **kwargs: None):
    """Get many code snippets from GPT-3 ordered from most to least likely"""

    # Beam search would be preferable, but it's computationally costly
    # (for OpenAI, which is why they don't offer it)

    while t <= 1:
        try:
            log_gpt_call(source=source, instruction=instruction, modality=modality,
                         n=batch_size, t=t)
            yield from query_gpt(source=source, instruction=instruction, modality=modality,
                                 n=batch_size, t=t)
        except openai.error.Timeout as e:
            logging.info(traceback.format_exc())
            pass
        except openai.error.InvalidRequestError as e:
            logging.error(traceback.format_exc())

            if token_error_message in e.error.message:
                logging.info('Stopping iterations due to token limit error')
                break

        t += delta_t


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


# TODO possibly use different classes with inheritance for different modes and explicit arguments
class CustomLLMChain:
    def __init__(
            self,
            temperature: float = 0.,
            mode: str = "generate",
            model_name: str = "codellama:7b-instruct"
    ):
        # TODO switch to OpenAI based on the model name
        self.temperature = temperature
        self.mode = mode
        self.model_name = model_name
        self.llm = self.create_chain()


    def create_chain(self) -> LLMChain:
        chat_prompt_template = create_chat_prompt_template(self.mode)
        if "gpt" in self.model_name.lower():
            chat_model = ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature,
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                openai_organization=os.getenv("OPENAI_ORG")
            )
        elif "llama" in model_name.lower():
            chat_model = ChatOllama(
                model=model_name,
                temperature=temperature
            )

        return LLMChain(llm=chat_model, prompt=chat_prompt_template)

    def run(self, **kwargs):
        return llm.run(**kwargs)


def query_llm(
        language: str,
        temperature: float = 0.,
        mode: str = "generate",
        model_name: str = "codellama:7b-instruct",
        **kwargs
) -> str:
    llm = CustomLLMChain(temperature=temperature, mode=mode, model_name=model_name)
    result = llm.run(**kwargs)

    if mode != "explain_bugs":
        result = postprocess_code(code=result, language=language)

    return result


def explore_llm(
        llm: LLMChain,
        language: str,
        log_gpt_call: Callable,
        mode: str = "generate",
        model_name: str = "codellama:7b-instruct",
        t: float = 0.0,
        delta_t: float = 0.2,
        **kwargs
) -> Generator[str]:
    while t <= 1:
        log_gpt_call(**locals())
        yield from query_llm(
            language=language,
            temperature=t,
            mode=mode,
            model_name=model_name,
            **kwargs
        )

        t += delta_t



if __name__ == '__main__':
    import itertools
    logging.basicConfig(level=logging.INFO)

    for code in itertools.islice(explore_gpt(instruction=input()), 10):
        print(code)
