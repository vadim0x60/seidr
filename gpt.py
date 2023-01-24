import logging
import openai
from tenacity import retry, retry_if_exception_type, stop_after_attempt
from tenacity import wait_random_exponential
import traceback

token_error_message = 'tokens for the input and instruction but the maximum allowed is 3000. ' \
                      'Please reduce the input or instruction length.'


@retry(retry=retry_if_exception_type(openai.error.APIError) |
             retry_if_exception_type(openai.error.APIConnectionError) |
             retry_if_exception_type(openai.error.ServiceUnavailableError),
       wait=wait_random_exponential(max=300),
       stop=stop_after_attempt(50))
@retry(retry=retry_if_exception_type(openai.error.RateLimitError),
       wait=wait_random_exponential(max=600))
def query_gpt(source=None, instruction=None, modality='code', n=1, temperature=1.0):
    """
    Get code snippets from GPT-3.

    If instruction is not specified, the code is extended (autocompleted),
    otherwise it's edited according to the instruction.
    """
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
                temperature=temperature
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
                temperature=temperature
            )

            if modality == 'text':
                result = [choice['text'] for choice in response["choices"]
                          if "text" in choice.keys()]
                logging.info(f"\nBug summary by GPT:\n{result[0]}\n")
            else:
                result = [source + '\n' + choice['text'] for choice in response["choices"]
                          if "text" in choice.keys()]

    except openai.error.InvalidRequestError as e:
        result = []
        if token_error_message in e.error.message:
            raise e

def explore_gpt(source='', instruction=None, modality='code', batch_size=1, heat_per_batch=0.2):
    """Get many code snippets from GPT-3 ordered from most to least likely"""

    # Beam search would be preferable, but it's computationally costly
    # (for OpenAI, which is why they don't offer it)

    # We fix moderate temperature to get sufficiently varied code snippets from the model
    temperature = 0.0

    while temperature <= 1:
        # We intentionally avoid temperature=0
        # That would lead to a batch of identical code snippets
        # Update temperature but keep it 1 at max
        temperature += heat_per_batch

        try:
            yield from query_gpt(source=source, instruction=instruction, modality=modality,
                                 n=batch_size, temperature=temperature)
        except openai.error.InvalidRequestError as e:
            logging.error(traceback.format_exc())

            if token_error_message in e.error.message:
                logging.info('Stopping iterations due to token limit error')
                break

if __name__ == '__main__':
    import itertools

    for code in itertools.islice(explore_gpt(input()), 10):
        print(code)
