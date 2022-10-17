# Tests for project 1

We provide some tests for your submission. Clone this repo and run them from this directory:

```
pytest --github_link <GITHUB-REPO-URL> .
```

We assume that the root of your repository contains a `README.md` file, a script `run.py` (or `run.ipynb`) and a file `implementations.py`  containing the functions with the requested signatures.
To iterate faster on the tests, you can give a local directory instead of your `<GITHUB-REPO-URL>` (ignore the Github URL failing tests).

### Environment

The tests will be run in a conda environement that you can create as follow:

```
conda create --file=environment.yml --name=project1-grading
conda activate project1-grading
```

### Submission

This is a standalone test suite meant to be run from outside your repo. 
**You should not copy these test files** to your repo as they could be updated.

Before submitting your Github link, you should:
- pull the latest version of these tests,
- run these tests on the Github link from a computer that has pull access to your private repository.

*Note that the public tests are meant to check your function signatures. Passing them does not guarantee that you will have 100% credit for the coding part.*

### Formatting (optional)

We advise you to format your code with the [black formatter](https://github.com/psf/black):

```
black <SOURCE-DIRECTORY>
```