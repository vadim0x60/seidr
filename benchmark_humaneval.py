"""Load and benchmark SEIDR on HumanEval-x"""

import logging
import os
import pathlib
import random
import traceback
import wandb
from pathlib import Path
from typing import List, Optional, Tuple

from fire import Fire
from programlib import Program, Language
from programlib import language_

from parse_humaneval_tests import load_jsonl
from seidr import get_template
from seidr.dev import SEIDR
from seidr.eval import UnitTest
from seidr.github import FileLogger

logger = logging.getLogger(__name__)

DATA_PATH = os.environ.get("DATA_PATH")

debug_templates = [
    line.split("\t") for line in get_template("prompts.txt").splitlines()
]
debug_templates = {int(ix.strip()): prompt.strip() for ix, prompt in debug_templates}


def is_already_solved(
    solutions_logger: FileLogger,
    test_data: Tuple[List[str] | str, List[str] | str],
    language: Language,
) -> Program | bool:
    """Checks if the currently logged solution passes all tests in `test_data`.
    Returns False if a Program class instance cannot be created"""
    try:
        return Program(
            workdir=solutions_logger.dir,
            name=solutions_logger.filename,
            language=language,
        ).test(test_data)
    except FileNotFoundError:
        return False


def load_humaneval_problem(
    data_path: pathlib.Path | str, language: str = "Python", problem: str = "Python/0"
) -> (str, List[str], str):
    """Load prompt, tests and the canonical solution from parsed tests
    from `data_path` path to the `humaneval` folder with jsonl files"""
    language = "cpp" if language.lower() == "c++" else language.lower()
    data = load_jsonl(data_path / f"humaneval_{language}_split_tests.jsonl")
    task = [item for item in data if item["task_id"] == problem][0]
    return task["prompt"], task["tests_split"], task["canonical_solution"]


def run_benchmark(
    problem: str = "fizz-buzz",
    language: str = "C++",
    max_programs: int = 1000,
    drafts_per_prompt: int = 10,
    explanations_per_program: int = 10,
    repairs_per_explanation: int = 2,
    beam_width: int = 100,
    seed: int = 42,
    valid_examples: int = 100,
    test_examples: int = 2000,
    prompt_examples: int = 5,
    log: str = "ERROR",
    model_name: str = "gpt-3.5-turbo",
    lexicase_selection: bool = False,
    ollama_url: Optional[str] = "http://localhost:11434",
    experiment_id: int = 0,
    **kwargs,
):
    """Generate and repair programs in HumanEval (C++ and Python)

    Parameters
    ----------
    problem : str
        name of a problem in PSB 2
    language : str
        programming language
    max_programs : int
        maximum number of elements in the resulting beam search tree
    drafts_per_prompt : int
        number of drafted problem solutions to be generated from a prompt
    explanations_per_program : int
        number of natural language explanations to give for one program (that does not pass all validation tests)
    repairs_per_explanation : int
        number of debugging attempts for each error explanation
    beam_width : int
        number of elements with top score that will be kept in the beam
        out of all leaves at the newly created level n+1
    seed : int
        used to fix seed so that the same I/O pairs are fetched from PSB2
    valid_examples : int
        number of I/O pairs to fetch as train set for program generation,
        upper limit: 1M (set by PSB2), edge cases appear first,
        random pairs are added to fill up the required number
    test_examples : int
        number of I/O pairs to fetch as test set for program generation,
        upper limit: 1M (set by PSB2),
        test cases contain only random I/O pairs, but no edge cases.
    prompt_examples : int
        number of I/O pairs taken from n_train_pairs to generate initial prompt
        for Codex completion model
    log : str
        logging mode, mostly used INFO, ERROR or DEBUG in our experiments
    model_name : str
        name of the OpenAI or Ollama model to use
    lexicase_selection : bool
        whether to use lexicase selection or just sort by score
    ollama_url : str
        link to the ollama cluster, default is localhost
    """
    # Setup logging
    Path("logs").mkdir(exist_ok=True)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=log.upper(),
    )
    logging.info("logging info")

    config = {
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_task_pid": os.environ.get("SLURM_TASK_PID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "task_id": os.environ.get("TASK_ID"),
        **kwargs,
        **locals(),
    }

    del config["kwargs"]
    model_name_tag = model_name.replace(":", "_")
    run = wandb.init(
        entity=os.environ.get("WANDB_ENTITY"),
        project=f"seidr-telo-humaneval-{model_name_tag}-run{experiment_id}",
        dir=os.environ.get("WANDB_DIR"),
        config=config,
    )
    logger.info(f"Run config {run.config}, W&B: {run.url}")

    language = language_(language)

    commit_msg_template = get_template("commit.txt").format(
        problem=problem, wandb_url=run.url
    )

    lexicase_tag = "_lexicase" if lexicase_selection else ""
    attempts_branch = f"humaneval_{model_name_tag}_{drafts_per_prompt}x{explanations_per_program}x{repairs_per_explanation}{lexicase_tag}_run{experiment_id}_dev"
    solutions_branch = f"humaneval_{model_name_tag}_{drafts_per_prompt}x{explanations_per_program}x{repairs_per_explanation}{lexicase_tag}_run{experiment_id}"

    attempts_logger = FileLogger(
        branch=attempts_branch,
        filename=language.source.format(name=problem),
        commit_msg_template=commit_msg_template,
    )
    solutions_logger = FileLogger(
        branch=solutions_branch,
        filename=language.source.format(name=problem),
        commit_msg_template=commit_msg_template,
    )

    description = (
        "Complete the following code given the task description and function signature."
    )

    # ensure that the same I/O pairs are fetched for every experiment
    random.seed(seed)

    start_prompt, tests, canonical_solution = load_humaneval_problem(
        pathlib.Path(DATA_PATH) / "humaneval", language.name, problem
    )

    prompt_data = tests[: min(prompt_examples, len(tests))]
    valid_data = tests
    test_data = tests

    if len(test_data) == 0:
        logging.info(
            "All tests are validation tests, setting final tests to be equal to validation tests"
        )
        test_data = valid_data

    if is_already_solved(solutions_logger, test_data, language):
        logging.info(f"{problem} is already solved, shutting down")
        return

    call_count = 0

    def log_llm_call(**kwargs):
        """Update and log the number of LLM calls"""
        nonlocal call_count
        wandb.log({"llm_calls": call_count})
        call_count += 1

    validation_critics = [
        lambda code: UnitTest(code, language, test) for test in valid_data
    ]

    seidr = SEIDR(
        task_name=problem,
        task_description=description,
        critics=validation_critics,
        model_name=model_name,
        language=language,
        beam_width=beam_width,
        drafts_per_prompt=drafts_per_prompt,
        explanations_per_program=explanations_per_program,
        repairs_per_explanation=repairs_per_explanation,
        lexicase_selection=lexicase_selection,
        log_metrics=wandb.log,
        log_attempt=attempts_logger,
        log_solution=solutions_logger,
        log_llm_call=log_llm_call,
        max_programs=max_programs,
        ollama_url=ollama_url,
    )

    solution = seidr.develop(start_code=start_prompt)

    logging.info("Development done. Testing...")
    test_evals = [UnitTest(solution, language, test) for test in test_data]
    avg_score = sum(e.score() for e in test_evals) / len(test_evals)
    test_pass_rate = sum(e.check() for e in test_evals) / len(test_evals)

    logging.info(
        f"\nTest pass rate on test: {test_pass_rate}\nTest avg score on test: {avg_score}"
    )

    wandb.log({"test_avg_score": avg_score, "test_pass_rate": test_pass_rate})
    run.finish()
    wandb.finish()


if __name__ == "__main__":
    try:
        Fire(run_benchmark)
    except:
        logging.error(traceback.format_exc())
        raise
