from git import GitError
from git import Repo
import shutil
import logging


def upload_file(repo, filename, message=None):
    if not message:
        message = f'added {filename}'

    repo.index.add(filename)
    repo.index.commit(message)
    repo.remotes.origin.pull()
    repo.remotes.origin.push()
    logging.info(f'Pushed updates to git. \nCommit message: {message}')


def ensure_repo(remote, path, branch=None):
    try:
        repo = Repo(path)

        if branch:
            repo.git.checkout(branch)
    except GitError:
        shutil.rmtree(path, ignore_errors=True)
        repo = Repo.clone_from(remote, path)

        if branch:
            branches = [ref.name for ref in repo.references]
            repo.git.fetch('--all')
            if f'{repo.remote().name}/{branch}' in branches:
                repo.git.checkout(branch)
            else:
                repo.git.checkout('-b' + branch)
                repo.git.add(repo.working_dir)
                # repo.git.commit(m=f'New branch {branch}')
                repo.git.push('--set-upstream', repo.remote().name, branch)

    return repo


def config_repo(dir, branch):
    try:
        import os
        repo = ensure_repo(os.environ['GITHUB_REMOTE'], dir, branch=branch)
        repo.config_writer().set_value('user', 'name', os.environ['GIT_USER']).release()
        repo.config_writer().set_value('user', 'email', os.environ['GIT_EMAIL']).release()
        return repo
    except (KeyError, GitError):
        return None
