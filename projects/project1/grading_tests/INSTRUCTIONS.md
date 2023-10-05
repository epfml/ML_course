# Tests for project 1

We provide some tests to check your submission. 
Clone the course repository and run them from this directory:

```
pytest --github_link <GITHUB-REPO-URL> .
```

We assume that the root of your repository contains a `README.md` file, a script `run.py` (or `run.ipynb`) and a file `implementations.py`  containing the functions with the requested signatures.
This command should be run from a computer that has the access right to clone your repository.
You might need to set up SSH Keys for git by following these [instructions](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent#generating-a-new-ssh-key).
To iterate faster on the tests, you can give a local directory instead of your `<GITHUB-REPO-URL>` (ignore the Github URL failing tests).

We advise you not to copy these test files inside your project repository but rather use them from the course repository.
We might update the public tests and you should check for new commits to pull before running the tests.

*Note that the public tests are meant to check your function signatures. Passing them does not guarantee that you will have 100% credit for the coding part.*

### Environment

The tests will be run in a conda environement created as follow:

```
conda env create --file=environment.yml --name=project1-grading
conda activate project1-grading
```

### Formatting (optional)

We advise you to format your code with the [black formatter](https://github.com/psf/black):

```
black <SOURCE-DIRECTORY>
```
