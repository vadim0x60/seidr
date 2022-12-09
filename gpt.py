import openai
from fire import Fire
from tenacity import retry, retry_if_exception_type, stop_after_attempt
from tenacity import wait_random_exponential, wait_fixed

@retry(retry=retry_if_exception_type(openai.error.RateLimitError), 
       wait=wait_random_exponential())
@retry(retry=retry_if_exception_type(openai.error.APIError),
       wait=wait_fixed(20),
       stop=stop_after_attempt(3))
def query_gpt(code, instruction=None, n=1, temperature=1.0):
    """
    Get code snippets from GPT-3. 
    
    If instruction is not specified, the code is extended (autocompleted),
    otherwise it's edited according to the instruction.
    """

    if instruction:
        response = openai.Edit.create(
            engine="code-davinci-edit-001",
            input=code,
            n=n,
            instruction=instruction,
            temperature=temperature
        ) 

        return [choice['text'] for choice in response["choices"]]
    else:
        response = openai.Completion.create(
            engine="code-davinci-001",
            prompt=code,
            n=n,
            temperature=temperature,
            #stop=["\"\"\""]
        )
        
        return [code + '\n' + choice['text'] for choice in response["choices"]]

def explore_gpt(code, instruction=None, batch_size=1, heat_per_batch=0.2):
    """Get many code snippets from GPT-3 ordered from most to least likely"""

    # Beam search would be preferable, but it's computationally costly
    # (for OpenAI, which is why they don't offer it)

    temperature = 0

    while True:
        # We intentionally avoid temperature=0 
        # That would lead to a batch of identical code snippets
        temperature += heat_per_batch
        yield from query_gpt(code, instruction, 
                             n=batch_size, temperature=temperature)

if __name__ == '__main__':
    import itertools

    for code in itertools.islice(explore_gpt(input()), 10):
        print(code)