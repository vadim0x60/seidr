# Synthesize Execute Instruct Debug Rank

A framework for AI-assisted program synthesis.
Given a problem description and some input-output examples, the framework generates a program that solves the problem.

## Paper

You can find an in-depth discussion of this tool, the philosophy it implements and its usage in our paper, [Fully Autonomous Programming with Large Language Models](https://dl.acm.org/doi/abs/10.1145/3583131.3590481). Consider citing it if you use SEIDR in your research.

## Usage

```
from seidr import develop
help(develop)
```

## Reproducing the experiments from our paper

The experiments are contained in `benchmark.py` and `benchmark_humaneval.py` files. When you run this file, the AI-generated programs are commited to a dedicated github repository, while the metrics (i.e. how many tests every program passes) will be logged in your [Weights and Biases](https://wandb.ai)

### Set up Weights and Biases

1. Create an account on [Weights and Biases](https://wandb.ai)
2. Install the [Weights and Biases](https://docs.wandb.com/library/install) library
3. Run `wandb login` and follow the instructions

### Set up a github repository

1. Go to [github](https://github.com), log in to the account that's going to push AI-generated code. Remember the $username and $email for that account.
2. Go [here](https://github.com/settings/tokens?type=beta) and generate an access $token
3. Set `GITHUB_USER` to "Bot" or whatever the name of the committer shall be
4. Set `GITHUB_EMAIL` to $email
5. Set `GITHUB_REMOTE` to https://$username:$token@github.com/$repo

Don't be fooled by the variable names, you can of course use a non-github git hosting.

### Set up language model access

It's 2023, we are not going to tell you which Large Language Model to use or whether to run it in the cloud or locally.
SEIDR runs on langchain and supports OpenAI and Ollama out of the box + any langchain-compatible model inference backend with a little coding.
Make sure you have Ollama server running or `OPENAI_API_KEY` environment variable set.

### Run the experiments

If you're using [slurm](https://slurm.schedmd.com/), the (template) files you need to `sbatch` are found in `example_scripts` folder. They will require some editing for your setup. If you don't use slurm, run `benchmark.py` directly.
