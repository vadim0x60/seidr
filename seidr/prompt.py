import logging

from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.chains import LLMChain
from pathlib import Path
from string import Template

from programlib import language_
from seidr import get_template

def initial_prompt(task_description, examples):
    prompt = task_description
    prompt += '\nFor example,'
    for sample_inputs, sample_outputs in examples:
        prompt += ''
        prompt += '\ninput:\n'
        for sample_input in sample_inputs:
            prompt += sample_input + '\n'
        prompt += 'output:'
        for sample_output in sample_outputs:
            prompt += '\n' + sample_output
    return prompt


def write_debug_prompt(mistake, debug_prompt_text, task_description=None):
    logging.info('Updating debug prompt')

def llm_assisted_bug_summary(debug_prompt_text, task_description, input, expected_output, actual_output):
    return explore_llm(
        language=language,  # TODO get it from somewhere
        tempearture=t,  # TODO what to do with delta_t
        mode="explain_bugs",
        model_name="codellama:7b-instruct",  # TODO get it from somewhere
        task_name=task_name,  # TODO get it from somewhere
        code=code, # TODO get it from somewhere
        input=input,
        output=expected_output,
        wrong_output=actual_output,
        task_description=task_description
    )


def write_debug_prompt(mistake, debug_prompt_text, task_description=None):
    logging.info('Formation of debug prompt')
    output_lines = '\n'.join([s.decode("utf-8") if type(s) == bytes else s for s in mistake.output_lines])

    prompt = ""
    if mistake.correctness < 1:
        if mistake.exit_status:
            error_lines = output_lines
            prompt = f'Fix {error_lines}' # TODO: return bug summary instead of a prompt

        else:
            i = '\\n'.join(mistake.input_lines)
            o = '\\n'.join(mistake.expected_output_lines)
            if 'GPT ---' in debug_prompt_text: # TODO update to always use it
                
                prompt = gpt_assisted_prompt(
                    debug_prompt_text, task_description, mistake.input_lines,
                    mistake.expected_output_lines, output_lines)
            else:
                prompt = debug_prompt_text.format(i=i, o=o)
    else:
        logging.info('\n\nrun.correctness = 1 for all runs, mistake lines are empty\n\n')
        prompt = dont_change

    logging.info(f'The prompt is: \n{prompt}')
    return prompt


def start_coding(prompt, language='C++'):
    language = language_(language)
    template = language.source.format(name='template')
    template = get_template(template)
    template = Template(template)
    return template.substitute(prompt=prompt)


SYSTEM_PROMPT = """You are an experienced software developer. \
You write concise code in {language}. \
The code must read input from user and return output corresponding to the task description."""

HUMAN_PROMPTS = {
    "generate": "Solve the following code contest problem: {task_name}. "
                "Problem description: {task_description}.\n"
                "{start_code}\n"
                "Only complete the code, do not add triple quotes, do not give explanations.",
    "explain_bugs": "I'm trying to solve the following code contest problem: {task_name}. "
                    "Problem description: {task_description}.\n"
                    "Currently, the code is \n```\n{code}\n``` \n"
                    "The issue is \n"
                    "{issue}\n"
                    "Describe how I should fix the code in a very concise manner.",
    "repair": "Solve the following code contest problem: {task_name}. "
              "Problem description: {task_description}.\n"
              "Currently, the code is \n```\n{code}\n``` \n"
              "Modify the code as  {bug_summary}.\n"
              "You must only return correct code. Remove any triple quotes, language name or explanations. ",
    "format": "Currently, the code is \n```\n{code}\n``` \n"
              "Keep the current code imports and "
              "add the following header if it is not present: \n{first_lines}\n"
              "Remove triple quotes from the answer. Add input reading from the user."
              "Print output as in the examples, do not print other words."
}

class ModeNameError(KeyError):
    """Used to catch a wrong human prompt key"""
    pass


def create_chat_prompt_template(
        mode: str = "generate",
) -> ChatPromptTemplate:
    """Returns chat prompt template in the form of two messages, system prompt and human message,
    depending on the `mode`"""
    try:
        system_message_template = SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT)
        human_message_template = HumanMessagePromptTemplate.from_template(HUMAN_PROMPTS[mode])

        chat_prompt_template = ChatPromptTemplate.from_messages(
            [system_message_template, human_message_template]
        )
    except KeyError:
        raise ModeNameError(f"Unsupported mode name '{mode}'. Use one of {HUMAN_PROMPTS.keys()}")

    return chat_prompt_template


