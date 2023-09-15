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

### Set up OpenAI access

It's 2022 and the language model inference happens in the cloud.
You are going to need an OpenAI account with access to `code-davinci-001` and `code-davinci-edit-001`
Set `OPENAI_API_KEY` environment variable to your access token.

### Run the experiments

If you're using [slurm](https://slurm.schedmd.com/), write a `run.sh` file with `python benchmark.py` and run it with `sbatch run.sh --array=0-191`.
If not, run `TASK_ID=n python benchmark.py` to re-run one of our 192 experiments exactly, or set the parameters yourself:

```
python benchmark.py --branching-factor 200 --language C++ --problem fizz-buzz
```
