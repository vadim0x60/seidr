from git import GitCommandError, Repo

def upload_file(repo, filename, message=None):
    if not message:
        message = f'added {filename}'

    repo.index.add(filename)
    repo.index.commit(message)
    repo.remotes.origin.pull()
    repo.remotes.origin.push()

def ensure_repo(remote, path):
    try:
        repo = Repo.clone_from(remote, path)
    except GitCommandError:
        repo = Repo(path)
    assert repo.remotes.origin.url == remote
    return repo