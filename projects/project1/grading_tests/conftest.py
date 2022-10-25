import importlib.machinery
import pathlib
import shutil
import sys

import git  # pip install gitpython
import pytest

RTOL = 1e-4
ATOL = 1e-8


GITHUB_LINK = None


def pytest_addoption(parser):
    parser.addoption("--github_link", action="store", required=True)


def pytest_configure(config):
    global GITHUB_LINK
    GITHUB_LINK = config.option.github_link


@pytest.fixture(scope="session")
def github_repo_path() -> pathlib.Path:
    workdir = pathlib.Path("github_workdir").resolve()

    # Support giving a directory path instead of github link
    if not GITHUB_LINK.startswith("https://"):
        yield pathlib.Path(GITHUB_LINK)
    else:
        if GITHUB_LINK.split("/")[-2] == "tree":
            parts = GITHUB_LINK.split("/")
            commit = parts[-1]
            url = "/".join(parts[:-2]) + ".git"
        else:
            commit = None
            url = GITHUB_LINK

        repo = git.Repo.clone_from(url, to_path=workdir)

        if commit is not None:
            submitted_branch = repo.create_head("submitted", commit)
            repo.head.reference = submitted_branch
            assert not repo.head.is_detached
            # reset the index and working tree to match the pointed-to commit
            repo.head.reset(index=True, working_tree=True)

        yield workdir
        shutil.rmtree(workdir, ignore_errors=True)


@pytest.fixture(scope="session")
def student_implementations(github_repo_path: pathlib.Path):
    sys.path.insert(0, str(github_repo_path.resolve()))
    loader = importlib.machinery.SourceFileLoader(
        "student_implementations", str(github_repo_path / "implementations.py")
    )
    handle = loader.load_module("student_implementations")
    return handle
