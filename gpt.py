import logging

import openai
from tenacity import retry, retry_if_exception_type, stop_after_attempt
from tenacity import wait_random_exponential, wait_fixed

token_error_message = 'tokens for the input and instruction but the maximum allowed is 3000. ' \
                      'Please reduce the input or instruction length.'


@retry(retry=retry_if_exception_type(openai.error.APIError) |
             retry_if_exception_type(openai.error.APIConnectionError) |
             retry_if_exception_type(openai.error.ServiceUnavailableError),
       wait=wait_random_exponential(max=300),
       stop=stop_after_attempt(5))
@retry(retry=retry_if_exception_type(openai.error.RateLimitError),
       wait=wait_random_exponential(max=600))
def query_gpt(code, instruction=None, code_behavior=None, n=1, temperature=1.0):
    """
    Get code snippets from GPT-3. 
    
    If instruction is not specified, the code is extended (autocompleted),
    otherwise it's edited according to the instruction.
    """
    try:
        if code_behavior:
            logging.info("Calling GPT for bug summarization")
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=code_behavior,
                n=n,
                temperature=temperature
            )
            result = [choice['text'] for choice in response["choices"] if "text" in choice.keys()]
            logging.info(f"\nBug summary by GPT:\n{result[0]}\n")
        elif code:
            if instruction:
                logging.info(f"Calling Codex-edit to debug code with instruction \n{instruction}")
                response = openai.Edit.create(
                    engine="code-davinci-edit-001",
                    input=code,
                    n=n,
                    instruction=instruction,
                    temperature=temperature
                )
                result = [choice['text'] for choice in response["choices"] if "text" in choice.keys()]
            else:
                logging.info(f"Calling Codex-completion to create initial program from template")
                logging.debug(f'template: \n{code}')
                response = openai.Completion.create(
                    engine="code-davinci-002",
                    prompt=code,
                    n=n,
                    temperature=temperature,
                )
                result = [code + '\n' + choice['text'] for choice in response["choices"] if "text" in choice.keys()]
                if len(result) == 0:
                    result = [code]
    except openai.error.InvalidRequestError as e:
        result = []

        if token_error_message in e.error.message:
            raise e
    return result


def explore_gpt(code='', instruction=None, code_behavior=None, batch_size=1, heat_per_batch=0.2):
    """Get many code snippets from GPT-3 ordered from most to least likely"""

    # Beam search would be preferable, but it's computationally costly
    # (for OpenAI, which is why they don't offer it)

    # We fix moderate temperature to get sufficiently varied code snippets from the model
    temperature = 0.0

    while temperature <= 1:
        # We intentionally avoid temperature=0
        # That would lead to a batch of identical code snippets
        # Update temperature but keep it 1 at max
        temperature = temperature + heat_per_batch

        try:
            yield from query_gpt(code, instruction, code_behavior,
                                 n=batch_size, temperature=temperature)
        except openai.error.InvalidRequestError as e:
            if token_error_message in e.error.message:
                logging.info('Stopping iterations due to token limit error')
                break


if __name__ == '__main__':
    import itertools

    for code in itertools.islice(explore_gpt(input()), 10):
        print(code)
