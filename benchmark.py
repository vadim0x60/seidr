import logging
import os
import random
import traceback
import pandas as pd
import psb2
import wandb
from seidr import get_template
from seidr.dev import develop
from seidr.eval import IOMatch, UnitTest
from seidr.github import FileLogger
from fire import Fire
from more_itertools import chunked
from programlib import Program
from programlib import language_
from pathlib import Path

from seidr.prompt import start_coding, initial_prompt

logger = logging.getLogger(__name__)

DATA_PATH = os.environ.get('DATA_PATH') or 'psb2'

task_descriptions = []
with open('psb2-meta/tasks.txt') as f:
    task_descriptions = {name.strip(): description.strip()
                         for name, description in chunked(f.readlines(), 2)}

debug_templates = [line.split('\t')
                   for line in get_template('prompts.txt').splitlines()]
debug_templates = {int(ix.strip()): prompt.strip()
                   for ix, prompt in debug_templates}


def title2kebabcase(title):
    return '-'.join(word.lower() for word in title.split(' '))


pushgp_success_rates = pd.read_csv('psb2-meta/results.tsv',
                                   sep='\t', index_col=['Problem'])
pushgp_success_rates = pushgp_success_rates['Succ.'].rename(title2kebabcase)


def is_already_solved(solutions_logger, test_data, language):
    try:
        return Program(workdir=solutions_logger.dir,
                       name=solutions_logger.filename,
                       language=language).test(test_data)
    except FileNotFoundError:
        return False


def run_benchmark(problem='fizz-buzz', language='C++', branching_factor=100,
                  max_programs=1000, beam_width=100, debug_prompt_id=0,
                  seed=42, valid_examples=100, test_examples=2000,
                  prompt_examples=5, batch_size=10, mode='execute', log='ERROR',
                  lexicase_selection=False,
                  **kwargs):
    """Generate and repair programs in PSB2

    Parameters
    ----------
    problem : str
        name of a problem in PSB 2
    language : str
        programming language
    branching_factor : int
        number of leaves to create at level n+1
        from each current leaf at level n in the beam
    max_programs : int
        maximum number of elements in the resulting beam
    beam_width : int
        number of elements with top score that will be kept in the beam
        out of all leaves at the newly created level n+1
    debug_prompt_id : int
        prompt template id from `./debug-prompt-templates/prompts.txt
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
    batch_size : int
        number of Codex outputs for the same prompt that will be generated at once
        for one parent during the beam search
    mode : str
        'execute' or 'debug'
    """
    # Setup logging
    Path('logs').mkdir(exist_ok=True)
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=log.upper())
    logging.info('logging info')
    baseline = pushgp_success_rates[problem]

    config = {
        'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
        'task_id': os.environ.get('TASK_ID'),
        **kwargs,
        **locals()
    }

    del config['kwargs']
    run = wandb.init(entity=os.environ.get('WANDB_ENTITY'), project='codex-for-psb', config=config)
    logger.info(f'Run config {run.config}, W&B: {run.url}')

    language = language_(language)

    commit_msg_template = get_template('commit.txt').format(
        problem=problem,
        wandb_url=run.url)

    lexicase_tag = '_lexicase' if lexicase_selection else ""

    attempts_branch = f'psb2_bf{branching_factor}_promptid{debug_prompt_id}_maxprograms{max_programs}{lexicase_tag}_dev'
    solutions_branch = f'psb2_bf{branching_factor}_promptid{debug_prompt_id}_maxprograms{max_programs}{lexicase_tag}'

    attempts_logger = FileLogger(branch=attempts_branch,
                                 filename=language.source.format(name=problem),
                                 commit_msg_template=commit_msg_template)
    solutions_logger = FileLogger(branch=solutions_branch,
                                  filename=language.source.format(name=problem),
                                  commit_msg_template=commit_msg_template)

    description = task_descriptions[problem]
    debug_template = debug_templates[debug_prompt_id]

    # ensure that the same I/O pairs are fetched for every experiment
    random.seed(seed)

    train_data, test_data = psb2.fetch_examples(
        DATA_PATH, problem, max(valid_examples, prompt_examples),
        test_examples, format='competitive')
    prompt_data = train_data[:prompt_examples]
    valid_data = train_data[:valid_examples]

    if mode == 'debug':
        for ix in range(5):
            train_data, test_data = psb2.fetch_examples(DATA_PATH, problem, 5, 10, format='competitive')
            for filename, data in zip([f'train_{ix}.txt', f'test_{ix}.txt'], [train_data, test_data]):
                with open(Path('solutions') / filename, 'w') as f:
                    f.writelines(list(map(lambda x: '\t'.join([x[0][0], x[1][0]]) + '\n', data)))

    if is_already_solved(solutions_logger, test_data, language):
        logging.info(f'{problem} is already solved, shutting down')
        return

    call_count = 0

    def log_gpt_call(**kwargs):
        nonlocal call_count
        wandb.log({'gpt_calls': call_count})
        call_count += 1

    critics = [
        lambda code: IOMatch(code, language=language, input=inp, output=out,
                             debug_template=debug_template,
                             task_description=description)
        for inp, out in valid_data
    ]
    prompt = initial_prompt(description, prompt_data)
    start_prompt = start_coding(prompt, language=language)

    solution = develop(task_description=description,
                       start_prompt=start_prompt,
                       critics=critics,
                       language=language,
                       beam_width=beam_width,
                       branching_factor=branching_factor,
                       lexicase_selection=lexicase_selection,
                       max_programs=max_programs,
                       log_metrics=wandb.log,
                       log_attempt=attempts_logger,
                       log_solution=solutions_logger,
                       log_gpt_call=log_gpt_call,
                       batch_size=min(batch_size, branching_factor))

    logging.info('Development done. Testing...')

    test_evals = [
        IOMatch(solution,
                language=language,
                input=inp, output=out,
                debug_template=debug_template,
                task_description=description)
        for inp, out in test_data]
    avg_score = sum(e.score() for e in test_evals) / len(test_evals)
    test_pass_rate = sum(e.check() for e in test_evals) / len(test_evals)

    wandb.log({'test_avg_score': avg_score,
               'test_pass_rate': test_pass_rate})
    run.finish()


if __name__ == '__main__':
    try:
        Fire(run_benchmark)
    except:
        logging.error(traceback.format_exc())
        raise
