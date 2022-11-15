from git import InvalidGitRepositoryError, GitCommandError, NoSuchPathError
from git import Repo
import shutil

def upload_file(repo, filename, message=None):
    if not message:
        message = f'added {filename}'

    repo.index.add(filename)
    repo.index.commit(message)
    repo.remotes.origin.pull()
    repo.remotes.origin.push()

def ensure_repo(remote, path, branch=None):
    try:
        repo = Repo(path)

        if branch:
            repo.git.checkout(branch)
    except (InvalidGitRepositoryError, GitCommandError, NoSuchPathError):
        shutil.rmtree(path, ignore_errors=True)
        repo = Repo.clone_from(remote, path)

        if branch:
            repo.git.checkout(branch)

    return repo