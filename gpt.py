import openai
from tenacity import retry, retry_if_exception_type, stop_after_attempt
from tenacity import wait_random_exponential
import traceback

@retry(retry=retry_if_exception_type(openai.error.RateLimitError),
       wait=wait_random_exponential())
@retry(retry=retry_if_exception_type((openai.error.APIError, 
                                      openai.error.APIConnectionError,
                                      openai.error.ServiceUnavailableError)),
       wait=wait_random_exponential(),
       stop=stop_after_attempt(50))
def query_gpt(source=None, instruction=None, modality='code', n=1, temperature=1.0):
    """
    Get code snippets from GPT-3. 
    
    If instruction is not specified, the code is extended (autocompleted),
    otherwise it's edited according to the instruction.
    """
    if instruction:
        if modality == 'text':
            engine = "text-davinci-edit-003"
        elif modality == 'code':
            engine = "code-davinci-edit-001" 
        else:
            raise ValueError(f'Unknown modality: {modality}')   

        response = openai.Edit.create(
            engine=engine,
            input=source,
            n=n,
            instruction=instruction,
            temperature=temperature
        )
        return [choice['text'] for choice in response["choices"] 
                if "text" in choice.keys()]
    else:
        if modality == 'text':
            engine = "text-davinci-003"
        elif modality == 'code':
            engine = "code-davinci-002" 
        else:
            raise ValueError(f'Unknown modality: {modality}')   

        response = openai.Completion.create(
            engine=engine,
            prompt=source,
            n=n,
            temperature=temperature
        )
        return [code + '\n' + choice['text'] for choice in response["choices"] 
                if "text" in choice.keys()]

def explore_gpt(batch_size=1, heat_per_batch=0.2, **kwargs):
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

        try:
            yield from query_gpt(**kwargs, n=batch_size, temperature=temperature)
        except openai.error.InvalidRequestError:
            print(locals())
            traceback.print_exc()
            pass

if __name__ == '__main__':
    import itertools

    for code in itertools.islice(explore_gpt(input()), 10):
        print(code)