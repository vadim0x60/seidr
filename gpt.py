import openai
from tenacity import retry, retry_if_exception_type, stop_after_attempt
from tenacity import wait_random_exponential

@retry(retry=retry_if_exception_type(openai.error.RateLimitError),
       wait=wait_random_exponential())
@retry(retry=retry_if_exception_type((openai.error.APIError, 
                                      openai.error.InvalidRequestError, 
                                      openai.error.APIConnectionError)),
       wait=wait_random_exponential(),
       stop=stop_after_attempt(50))
def query_gpt(code=None, code_behavior=None, instruction=None, n=1, temperature=1.0):
    """
    Get code snippets from GPT-3. 
    
    If instruction is not specified, the code is extended (autocompleted),
    otherwise it's edited according to the instruction.
    """
    if code_behavior:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=code_behavior,
            n=n,
            temperature=temperature
        )
        result = [choice['text'] for choice in response["choices"] if "text" in choice.keys()]
        return result if len(result) > 0 else []
    elif code:
        if instruction:
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
            response = openai.Completion.create(
                engine="code-davinci-001",
                prompt=code,
                n=n,
                temperature=temperature,
            )
            result = [code + '\n' + choice['text'] for choice in response["choices"] if "text" in choice.keys()]
            return result if len(result) > 0 else [code]

def explore_gpt(code='', instruction=None, code_behavior=None, batch_size=1, heat_per_batch=0.2):
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
            if 1.0 - temperature >= heat_per_batch else temperature

        yield from query_gpt(code, instruction, code_behavior,
                             n=batch_size, temperature=temperature)

if __name__ == '__main__':
    import itertools

    for code in itertools.islice(explore_gpt(input()), 10):
        print(code)