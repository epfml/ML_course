We provide some tests for your submission. Run them in this directory:

```
pytest --github_link <GITHUB-REPO-URL> .
```

We assume that the root of your repository contains a `README.md` file, a script `run.py` (or `run.ipynb`) and the functions with the requested signatures in `implementations.py`.
To iterate faster on the tests, you can give a local directory instead of your `<GITHUB-REPO-URL>`.
However, you should test the Github link that you will submit at the end.

We advise you to format your code with the [black formatter](https://github.com/psf/black):

```
pip install black
black <SOURCE-DIRECTORY>
```