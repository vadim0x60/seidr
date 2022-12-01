import logging
import os
import psb2
import pandas as pd
import random
import time
import wandb

from fire import Fire
from more_itertools import chunked
from pathlib import Path
from uuid import uuid4
from programlib import language_

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
                  max_tries=1000, beam_width=100, debug_prompt_id=0,
                  seed=42, n_train_pairs=2000, n_test_pairs=2000,
                  n_pairs_in_prompt=5, mode='execute'):
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
    max_tries : int
        maximum number of elements in the resulting beam
    beam_width : int
        number of elements with top score that will be kept in the beam
        out of all leaves at the newly created level n+1
    debug_prompt_id : int
        prompt template id from `./debug-prompt-templates/prompts.txt
    seed : int
        used to fix seed so that the same I/O pairs are fetched from PSB2
    n_train_pairs : int
        number of I/O pairs to fetch as train set for program generation,
        upper limit: 1M (set by PSB2), edge cases appear first,
        random pairs are added to fill up the required number
    n_test_pairs : int
        number of I/O pairs to fetch as test set for program generation,
        upper limit: 1M (set by PSB2),
        test cases contain only random I/O pairs, but no edge cases.
    n_pairs_in_prompt : int
        number of I/O pairs taken from n_train_pairs to generate initial prompt
        for Codex completion model
    mode : str
        'execute' or 'debug'

    """
    # Setup logging
    Path('logs').mkdir(exist_ok=True)
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO,
                        filename=f'./logs/run_bf{branching_factor}_promptid{debug_prompt_id}.log')

    baseline = pushgp_success_rates[problem]

    # Passing locals() as parameter to avoid RecursionError in wandb
    # when config = locals() as a variable, config contains itself recursively
    run = wandb.init(project='nl2ml-codex', config=locals())

    language = language_(language)
    os.makedirs('solutions', exist_ok=True)
    solutions_dir = Path('solutions') / str(uuid4())

    os.makedirs(solutions_dir, exist_ok=True)

    solutions_repo = ensure_repo(os.environ['GITHUB_REMOTE'], solutions_dir,
                                 branch=f'bf{branching_factor}_promptid{debug_prompt_id}')
    solutions_repo.config_writer().set_value('user', 'name', os.environ['GIT_USER']).release()
    solutions_repo.config_writer().set_value('user', 'email', os.environ['GIT_EMAIL']).release()

    description = task_descriptions[problem]
    debug_prompt_text = debug_templates[debug_prompt_id]

    # ensure that the same I/O pairs are fetched for every experiment
    random.seed(seed)
    train_data, test_data = psb2.fetch_examples(DATA_PATH, problem, n_train_pairs, n_test_pairs, format='competitive')

    if mode == 'debug':
        for ix in range(5):
            train_data, test_data = psb2.fetch_examples(DATA_PATH, problem, 5, 10, format='competitive')
            for filename, data in zip([f'train_{ix}.txt', f'test_{ix}.txt'], [train_data, test_data]):
                with open(solutions_dir / filename, 'w') as f:
                    f.writelines(list(map(lambda x: '\t'.join([x[0][0], x[1][0]]) + '\n', data)))

    solutionogen = develop(problem, description, train_data, n_pairs_in_prompt,
                           beam_depth=max_tries,
                           debug_prompt_text=debug_prompt_text, language=language,
                           beam_width=beam_width, branching_factor=branching_factor,
                           log_f=wandb.log)

    for idx, solution in enumerate(solutionogen):
        solution.test(test_data)
        wandb.log({'test_avg_score': solution.avg_score,
                   'test_pass_rate': solution.pass_rate,
                   'step_beam_search': idx})

        filename = language.source.format(name=problem)
        filepath = solutions_dir / filename
        solution.save(filepath)
        while not filepath.exists():
            time.sleep(1)
        upload_file(solutions_repo, filename, f'solution {idx} of {problem}, {solution.pass_rate} of tests passed')

        if idx >= max_tries:
            break

    run.finish()


experiments = [
    {'problem': problem, 
     'language': language, 
     'branching_factor': branching_factor, 
     'max_tries': 1000, 
     'beam_size': branching_factor}
    for problem in task_descriptions.keys()
    for language in ('C++', 'Python')
    for branching_factor in (1, 10, 100, 1000)
]

if __name__ == '__main__':
    task_id = os.environ.get('TASK_ID') or os.environ.get('SLURM_ARRAY_TASK_ID')
    logger.info('Start')
    if task_id is not None:
        run_benchmark(**experiments[int(task_id)])
    else:
        Fire(run_benchmark)
    logger.info('Finish')

