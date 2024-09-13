# Synthesize Execute Instruct Debug Rank

A framework for AI-assisted program synthesis.
Given a problem description and some input-output examples, the framework generates a program that solves the problem.

## Paper

You can find an in-depth discussion of this tool, the philosophy it implements and its usage in our paper, [Fully Autonomous Programming with Large Language Models](https://dl.acm.org/doi/abs/10.1145/3583131.3590481). Consider citing it if you use SEIDR in your research.

## Usage

```
from seidr import dev
help(dev)
```

## Reproducing the experiments from our paper

The experiments are contained in `benchmark.py` and `benchmark_humaneval.py` files. When you run this file, the AI-generated programs are commited to a dedicated github repository, while the metrics (i.e. how many tests every program passes) will be logged in your [Weights and Biases](https://wandb.ai)

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

OpenAI account is needed with access to `gpt-3.5-turbo` and 
an `OPENAI_API_KEY` environment variable 
set to your OpenAI API access token.


#### Set up Ollama

Run [Ollama](https://ollama.ai/) with Llama 3-8B or [another model](https://ollama.ai/library) locally 
or on a server. 
In the latter case, start the Ollama server with the following commands and note the `URL:PORT` pair:
```
OLLAMA_HOST=URL:PORT ollama serve &
OLLAMA_HOST=URL:PORT ollama pull llama3 &
```

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
If not, run `TASK_ID=n python benchmark.py` to re-run one of our experiments exactly, 
or set the parameters yourself as below.

For example, for basement problem in PSB2, run SEIDR without lexicase selection as follows:
```
python3 benchmark.py \
    --task_id 0 \
    --problem bowling \
    --language Python \
    --branching_factor 2 \
    --max_programs 100 \
    --drafts_per_prompt 2 \
    --explanations_per_program 2 \
    --repairs_per_explanation 2 \
    --beam_width 2 \
    --log INFO \
    --lexicase_selection False \
    --dataset humaneval \
    --model_name gpt-3.5-turbo \
    --valid_examples 50 \
    --experiment_id 0
```

To run an example with SEIDR with Llama 3 served by Ollama at `URL:PORT` on HumanEval with lexicase, run the following:
```
python3 benchmark_humaneval.py \
    --task_id 0 \
    --problem Python/0 \
    --language Python \
    --branching_factor 2 \
    --max_programs 100 \
    --drafts_per_prompt 2 \
    --explanations_per_program 2 \
    --repairs_per_explanation 2 \
    --beam_width 2 \
    --log INFO \
    --lexicase_selection True \
    --dataset humaneval \
    --model_name llama3 \
    --experiment_id 0 \
    --ollama_url "http://URL:PORT"

```

Example Slurm scripts are stored in `scripts/` and tables with hyperparameters in `/config`
