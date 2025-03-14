from string import Template
from typing import List, Tuple

from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
from programlib import language_

from seidr import get_template


def initial_prompt(task_description: str, examples: Tuple[List[str], List[str]]):
    """Create a docstring for the draft program (to be used in the generate mode)"""
    prompt = task_description
    prompt += "\nFor example,"
    for sample_inputs, sample_outputs in examples:
        prompt += ""
        prompt += "\ninput:\n"
        for sample_input in sample_inputs:
            prompt += sample_input + "\n"
        prompt += "output:"
        for sample_output in sample_outputs:
            prompt += "\n" + sample_output
    return prompt


def start_coding(prompt: str, language: str = "C++") -> str:
    """Fill in the template for the draft program with the first lines"""
    language = language_(language)
    template = language.source.format(name="template")
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
    "Print output as in the examples, do not print other words.",
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
        system_message_template = SystemMessagePromptTemplate.from_template(
            SYSTEM_PROMPT
        )
        human_message_template = HumanMessagePromptTemplate.from_template(
            HUMAN_PROMPTS[mode]
        )

        chat_prompt_template = ChatPromptTemplate.from_messages(
            [system_message_template, human_message_template]
        )
    except KeyError:
        raise ModeNameError(
            f"Unsupported mode name '{mode}'. Use one of {HUMAN_PROMPTS.keys()}"
        )

    return chat_prompt_template


def ollama_messages(mode: str = "generate", **kwargs) -> List[dict[str, str]]:
    """Returns messages formatted for interaction with ollama"""
    try:
        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT.format(**kwargs)},
            {"role": "user", "content": HUMAN_PROMPTS[mode].format(**kwargs)},
        ]
    except KeyError:
        raise ModeNameError(
            f"Unsupported mode name '{mode}'. Use one of {HUMAN_PROMPTS.keys()}"
        )

    return prompt
