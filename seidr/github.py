"""
Handy tools for recording our generated programs to a git repository.
May be nice to include this into programlib one day
"""

import git
import gitdb
from git import GitError
from git import Repo

from tenacity import retry, retry_if_exception_type, stop_after_attempt
from tenacity import wait_random_exponential
from programlib import Program

from pathlib import Path
from uuid import uuid4
import os
import shutil
import logging

@retry(retry=retry_if_exception_type(git.exc.GitCommandError) |
             retry_if_exception_type(git.exc.BadName) |
             retry_if_exception_type(gitdb.exc.BadName),
       wait=wait_random_exponential(),
       stop=stop_after_attempt(50))
def pullpush(repo):
    repo.remotes.origin.pull()
    repo.remotes.origin.push()


def upload_file(repo, filename, message=None):
    if not message:
        message = f'added {filename}'

    pullpush(repo)
    repo.index.add(filename)
    repo.index.commit(message)
    pullpush(repo)
    logging.info(f'Tried to push updates to git. \nCommit message: {message}')


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
                repo.git.push('--set-upstream', repo.remote().name, branch)

    return repo


def config_repo(dir, branch):
    try:
        import os
        logging.info('config_repo')
        repo = ensure_repo(os.environ['GIT_REMOTE'], dir, branch=branch)
        repo.config_writer().set_value('user', 'name', os.environ['GIT_USER']).release()
        repo.config_writer().set_value('user', 'email', os.environ['GIT_EMAIL']).release()
        repo.config_writer().set_value('pull', 'rebase', False).release()
        return repo
    except (KeyError, GitError) as e:
        logging.info(f'config_repo exception {e}')
        return None

class ProgramLogger:
    """
    A careful archivist keeping track of your program's versions.
    If git environment variables are set, will synchronize with a remote repository.
    """

    def __init__(self, branch, name, language, 
                 commit_msg_template='{message}'):
         os.makedirs('solutions', exist_ok=True)
         self.name = name
         self.language = language
         self.commit_msg_template = commit_msg_template
         cache_dir = branch + '_' + str(uuid4())[:6]
         self.dir = Path('solutions') / cache_dir
         self.repo = config_repo(self.dir, branch=branch)
         os.makedirs(self.dir, exist_ok=True)

    def current(self):
        return Program(workdir=self.dir, 
                       name=self.name, 
                       language=self.language)    

    def log(self, program, **vars):
        filename = program.language.source.format(name=self.name)
        program.save(self.dir / filename)
        if self.repo:
            commit_msg = self.commit_msg_template.format(**vars)
            upload_file(self.repo, filename, commit_msg)

    def __call__(self, *args, **kwargs):
        return self.log(*args, **kwargs)