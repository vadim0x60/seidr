"""Create a separate file with a list of experiments and their Slurm task id's"""

import logging
import pandas as pd
import traceback

from datetime import datetime
from pathlib import Path

from benchmark import task_descriptions


bf_experiments = [
    {'problem': problem,
     'language': language,
     'branching_factor': branching_factor,
     'max_programs': 100,
     'beam_width': branching_factor,
     'debug_prompt_id': 0,
     'log': 'INFO'}
    for problem in task_descriptions.keys()
    for language in ('C++', 'Python')
    for branching_factor in (2, 4, 16)
]

experiments = bf_experiments


if __name__ == '__main__':
    try:
        timestamp = datetime.now().strftime("%d_%m_%y__%H_%M")
        if not Path('config').exists():
            Path('config').mkdir()
        pd.DataFrame(experiments).to_csv(f'config/experiments_{timestamp}.csv')
    except:
        logging.error(traceback.format_exc())
        raise
