import logging

import openai
from tenacity import retry, retry_if_exception_type, stop_after_attempt
from tenacity import wait_random_exponential, wait_fixed


@retry(retry=retry_if_exception_type(openai.error.RateLimitError) |
             retry_if_exception_type(openai.error.ServiceUnavailableError) |
             retry_if_exception_type(openai.error.APIConnectionError),
       wait=wait_random_exponential())
@retry(retry=retry_if_exception_type(openai.error.InvalidRequestError) |
             retry_if_exception_type(openai.error.APIError),
       wait=wait_random_exponential(),
       stop=stop_after_attempt(3))
def query_gpt(code, instruction=None, code_behaviour=None, n=1, temperature=1.0):
    """
    Get code snippets from GPT-3. 
    
    If instruction is not specified, the code is extended (autocompleted),
    otherwise it's edited according to the instruction.
    """
    if code_behaviour:
        logging.info("Calling GPT for bug summarization")
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=code_behaviour,
            n=n,
            temperature=temperature
        )
        result = [choice['text'] for choice in response["choices"] if "text" in choice.keys()]
        logging.info(f"\nBug summary by GPT:\n{result[0]}\n")
        return result if len(result) > 0 else []
    elif instruction:
        logging.info(f"Calling Codex-edit to debug code with instruction \n{instruction}")
        response = openai.Edit.create(
            engine="code-davinci-edit-001",
            input=code,
            n=n,
            instruction=instruction,
            temperature=temperature
        )
        result = [choice['text'] for choice in response["choices"] if "text" in choice.keys()]
        return result if len(result) > 0 else []
    else:
        logging.info(f"Calling Codex-completion to create initial program from template")
        logging.debug(f'template: \n{code}')
        response = openai.Completion.create(
            engine="code-davinci-001",
            prompt=code,
            n=n,
            temperature=temperature,
        )
        result = [code + '\n' + choice['text'] for choice in response["choices"] if "text" in choice.keys()]
        return result if len(result) > 0 else [code]


def explore_gpt(code='', instruction=None, code_behaviour=None, batch_size=1, heat_per_batch=0.2):
    """Get many code snippets from GPT-3 ordered from most to least likely"""

    # Beam search would be preferable, but it's computationally costly
    # (for OpenAI, which is why they don't offer it)

    # We fix moderate temperature to get sufficiently varied code snippets from the model
    temperature = 0.0

    while True:
        # We intentionally avoid temperature=0 
        # That would lead to a batch of identical code snippets
        # Update temperature but keep it 1 at max
        temperature = temperature + heat_per_batch \
            if 1.0 - temperature > heat_per_batch else temperature

        yield from query_gpt(code, instruction=instruction, code_behaviour=code_behaviour,
                             n=batch_size, temperature=temperature)


if __name__ == '__main__':
    import itertools

    for code in itertools.islice(explore_gpt(input()), 10):
        print(code)
