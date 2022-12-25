from git import GitError
from git import Repo
import shutil

from tenacity import retry, retry_if_exception_type, stop_after_attempt
from tenacity import wait_random_exponential
from git.exc import GitCommandError

@retry(retry=retry_if_exception_type(GitCommandError),
       wait=wait_random_exponential(),
       stop=stop_after_attempt(50))
def pullpush(repo):
    repo.remotes.origin.pull()
    repo.remotes.origin.push()

def upload_file(repo, filename, message=None):
    if not message:
        message = f'added {filename}'

    repo.remotes.origin.pull()
    repo.index.add(filename)
    repo.index.commit(message)
    pullpush(repo)

def ensure_repo(remote, path, branch=None):
    try:
        repo = Repo(path)

        if branch:
            repo.git.checkout(branch)
    except GitError:
        shutil.rmtree(path, ignore_errors=True)
        repo = Repo.clone_from(remote, path)

        if branch:
            repo.git.checkout(branch)

    return repo

def config_repo(dir, branch):
    try:
        import os
        repo = ensure_repo(os.environ['GITHUB_REMOTE'], dir, branch=branch)
        repo.config_writer().set_value('user', 'name', os.environ['GIT_USER']).release()
        repo.config_writer().set_value('user', 'email', os.environ['GIT_EMAIL']).release()
        repo.config_writer().set_value('pull', 'rebase', False).release()
        return repo
    except (KeyError, GitError):
        return None