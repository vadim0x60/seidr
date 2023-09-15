from benchmark import run_benchmark, task_descriptions
import logging
from fire import Fire
import traceback

bf_experiments = [
    {'problem': problem, 
     'language': language, 
     'branching_factor': branching_factor, 
     'max_programs': 1000, 
     'beam_width': branching_factor}
    for problem in task_descriptions.keys()
    for language in ('C++', 'Python')
    for branching_factor in (1, 10, 100, 1000)
]

prompt_experiments = [
    {'problem': problem,
     'language': language,
     'branching_factor': branching_factor,
     'max_programs': 1000,
     'beam_width': branching_factor,
     'debug_prompt_id': debug_prompt_id,
     'batch_size': 10}
    for debug_prompt_id in range(11)
    for language in ('C++', 'Python')
    for problem in task_descriptions.keys()
    for branching_factor in [1]
]

experiments = bf_experiments + prompt_experiments

def run_experiment_for_paper(task_id, log='ERROR'):
    return run_benchmark(task_id=task_id, log=log, **experiments[task_id-1])

if __name__ == '__main__':
    try:
        Fire(run_experiment_for_paper)
    except:
        logging.error(traceback.format_exc())
        raise