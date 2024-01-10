# Synthesize Execute Instruct Debug Rank

A framework for AI-assisted program synthesis.
Given a problem description and some input-output examples, the framework generates a program that solves the problem.

## Usage

```
from seidr import develop
help(develop)
```

## Reproducing the experiments from our paper

The experiments reported in [the blog post](https://vadim.me/posts/unreasonable) and in the upcoming paper are contained in `benchmark.py` file. When you run this file, the AI-generated programs are commited to a dedicated github repository, while the metrics (i.e. how many tests every program passes) will be logged in your [Weights and Biases](https://wandb.ai)

### Prerequisites 
#### Set up Weights and Biases

1. Create an account on [Weights and Biases](https://wandb.ai)
2. Install the [Weights and Biases](https://docs.wandb.com/library/install) library
3. Run `wandb login` and follow the instructions

#### Set up a GitHub repository

1. Go to [github](https://github.com), log in to the account that's going to push AI-generated code. Remember the $username and $email for that account.
2. Go [here](https://github.com/settings/tokens?type=beta) and generate an access $token
3. Set `GIT_USER` to "Bot" or whatever the name of the committer shall be
4. Set `GIT_EMAIL` to $email
5. Set `GIT_REMOTE` to https://$username:$token@github.com/$repo

Note that you can use a non-GitHub git hosting.

#### Set up OpenAI access

OpenAI account is needed with access to `gpt-3.5-turbo`
Set `OPENAI_API_KEY` environment variable to your access token.

Example `.config` file layout:
```bash
# Github
export GIT_REMOTE=https://USERNAME:KEY@github.com/SOLUTIONS_REPO
export GIT_USER=...
export GIT_EMAIL=...

# Data
export DATA_PATH=...

# OpenAI
export OPENAI_API_KEY=...
export OPENAI_ORG=...

# WandB
export WANDB_ENTITY=...
export WANDB_DIR=...
```

### Run the experiments

If you're using [Slurm](https://slurm.schedmd.com/), write a `run.sh` file with `python benchmark.py` 
and run it with `sbatch run.sh --array=1-500`.
If not, run `TASK_ID=n python benchmark.py` to re-run one of our experiments exactly, or set the parameters yourself:

For example, for basement problem in PSB2, run SEIDR without lexicase selection as follows:
```
python3 benchmark.py \
    --task_id 202 \
    --problem basement \
    --language C++ \
    --max_programs 100 \
    --drafts_per_prompt 2 \
    --explanations_per_program 2 \
    --repairs_per_explanation 2 \
    --beam_width 2 \
    --log INFO \
    --lexicase_selection False \
    --dataset psb2 \
    --model_name gpt-3.5-turbo
```

Example Slurm scripts are stored in `example_scripts/` and tables with hyperparameters in `/config`