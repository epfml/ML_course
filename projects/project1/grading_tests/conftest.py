import importlib.machinery
import pathlib
import shutil
import sys

import git  # pip install gitpython
import pytest

RTOL = 1e-4
ATOL = 1e-8


GITHUB_LINK = None
CLONE_DIRECTORY = None
COMMIT_HASH = None
KEEP_REPO = False
USE_SSH = False


def pytest_addoption(parser):
    # fmt: off
    parser.addoption("--github_link", action="store", required=True)
    parser.addoption("--clone_directory", default="github_workdir", help="Directory to clone the repository into.")
    parser.addoption("--commit_hash", default=None, help="Specify a given commit to clone.")
    parser.addoption("--keep_repo", action="store_true", help="Do not delete the cloned repo after the tests.")
    parser.addoption("--use_ssh", action="store_true", help="Force using SSH authentication instead of password.")
    # fmt: on


def pytest_configure(config):
    global GITHUB_LINK
    GITHUB_LINK = config.option.github_link
    global CLONE_DIRECTORY
    CLONE_DIRECTORY = config.option.clone_directory
    global COMMIT_HASH
    COMMIT_HASH = config.option.commit_hash
    global KEEP_REPO
    KEEP_REPO = config.option.keep_repo
    global USE_SSH
    USE_SSH = config.option.use_ssh


@pytest.fixture(scope="session")
def github_repo_path() -> pathlib.Path:
    workdir = pathlib.Path(CLONE_DIRECTORY).resolve()

    # Support giving a directory path instead of github link
    if not GITHUB_LINK.startswith("https://") and not GITHUB_LINK.startswith("git@"):
        yield pathlib.Path(GITHUB_LINK)
    else:
        if GITHUB_LINK.split("/")[-2] in ["tree", "commit"]:
            parts = GITHUB_LINK.split("/")
            commit = parts[-1]
            url = "/".join(parts[:-2]) + ".git"
        else:
            commit = None
            url = GITHUB_LINK

        if COMMIT_HASH is not None:
            commit = COMMIT_HASH

        if USE_SSH and url.startswith("https://github.com/"):
            url = "git@github.com:" + url[len("https://github.com/") :]

        repo = git.Repo.clone_from(url, to_path=workdir)

        if commit is not None:
            submitted_branch = repo.create_head("submitted", commit)
            repo.head.reference = submitted_branch
            assert not repo.head.is_detached
            # reset the index and working tree to match the pointed-to commit
            repo.head.reset(index=True, working_tree=True)

        yield workdir
        if not KEEP_REPO:
            shutil.rmtree(workdir, ignore_errors=True)


@pytest.fixture(scope="session")
def student_implementations(github_repo_path: pathlib.Path):
    sys.path.insert(0, str(github_repo_path.resolve()))
    loader = importlib.machinery.SourceFileLoader(
        "student_implementations", str(github_repo_path / "implementations.py")
    )
    handle = loader.load_module("student_implementations")
    return handle
