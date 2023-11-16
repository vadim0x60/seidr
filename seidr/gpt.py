import logging
import openai
from tenacity import retry, retry_if_exception_type, stop_after_attempt
from tenacity import wait_random_exponential, before_sleep_log
import traceback

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

    source = str(source)
    instruction = str(instruction)
    n = int(n)
    t = float(t)

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

if __name__ == '__main__':
    import itertools
    logging.basicConfig(level=logging.INFO)

    for code in itertools.islice(explore_gpt(instruction=input()), 10):
        print(code)
