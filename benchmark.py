import logging
import os

import pandas as pd
import psb2
from fire import Fire
from more_itertools import chunked
from programlib import language_

import wandb
from develop import develop
from github import ensure_repo, upload_file

logger = logging.getLogger(__name__)

DATA_PATH = os.environ.get('DATA_PATH') or 'psb2'

with open('psb2-meta/tasks.txt') as f:
    task_descriptions = {name.strip(): description.strip() for name, description in chunked(f.readlines(), 2)}

with open('debug-prompt-templates/prompts.txt') as f:
    debug_templates = {int(ix.strip()): prompt.strip() \
                       for ix, prompt in list(map(lambda x: x.split('\t'), f.readlines()))}


def title2kebabcase(title):
    return '-'.join(word.lower() for word in title.split(' '))


pushgp_success_rates = pd.read_csv('psb2-meta/results.tsv', sep='\t', index_col=['Problem'])['Succ.'].rename(
    title2kebabcase)


def run_benchmark(problem, language='C++', branching_factor=100,
                  max_tries=1000, beam_size=100, debug_prompt_id=0):
    baseline = pushgp_success_rates[problem]

    config = locals()
    run = wandb.init(project='nl2ml-codex', config=config)

    language = language_(language)
    # solutions_repo = ensure_repo(os.environ['GITHUB_REMOTE'], 'solutions', branch=f'bf{branching_factor}')
    # solutions_repo.config_writer().set_value('user', 'name', os.environ['GIT_USER']).release()
    # solutions_repo.config_writer().set_value('user', 'email', os.environ['GIT_EMAIL']).release()

    description = task_descriptions[problem]
    debug_prompt_text = debug_templates[debug_prompt_id]
    train_data, test_data = psb2.fetch_examples(DATA_PATH, problem, 5, 2000, format='competitive')

    solutionogen = develop(problem, description, train_data,
                           debug_prompt_text=debug_prompt_text, language=language,
                           beam_size=beam_size, branching_factor=branching_factor,
                           log_f=wandb.log)

    for idx, solution in enumerate(solutionogen):
        solution.test(test_data)
        wandb.log({'test_avg_score': solution.avg_score,
                   'test_pass_rate': solution.pass_rate})

        filename = language.source.format(name=problem)
        solution.save('solutions/' + filename)
        # upload_file(solutions_repo, filename, f'solution {idx} of {problem}, {solution.pass_rate} of tests passed')

        if idx >= max_tries:
            break

    run.finish()


experiments = [
    lambda: run_benchmark(problem, language, branching_factor, 1000, branching_factor)
    for problem in task_descriptions.keys()
    for language in ('C++', 'Python')
    for branching_factor in (1, 10, 100, 1000)
]

if __name__ == '__main__':
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)

    task_id = os.environ.get('TASK_ID') or os.environ.get('SLURM_ARRAY_TASK_ID')
    logger.info('Start')
    if task_id is not None:
        experiments[int(task_id)]()
    else:
        Fire(run_benchmark)
