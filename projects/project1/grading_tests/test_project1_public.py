import pathlib

import numpy as np
import pytest

from conftest import ATOL, GITHUB_LINK, RTOL

FUNCTIONS = [
    "mean_squared_error_gd",
    "mean_squared_error_sgd",
    "least_squares",
    "ridge_regression",
    "logistic_regression",
    "reg_logistic_regression",
]


MAX_ITERS = 2
GAMMA = 0.1


@pytest.fixture()
def initial_w():
    return np.array([[0.5], [1.0]])


@pytest.fixture()
def y():
    return np.array([[0.1], [0.3], [0.5]])


@pytest.fixture()
def tx():
    return np.array([[2.3, 3.2], [1.0, 0.1], [1.4, 2.3]])


def test_github_link_format():
    assert GITHUB_LINK.startswith(
        "https://github.com/"
    ), "Please provide a Github link."
    assert (
        "tree" in GITHUB_LINK
    ), "Please provide a Github link ending with .../tree/... for the submission."


@pytest.mark.parametrize("filename", ("README.md", "implementations.py"))
def test_file_exists(filename: str, github_repo_path: pathlib.Path):
    assert (github_repo_path / filename).exists(), f"Missing file {filename}."


def test_run_script_exists(github_repo_path: pathlib.Path):
    if (
        not (github_repo_path / "run.py").exists()
        and not (github_repo_path / "run.ipynb").exists()
    ):
        raise FileNotFoundError("Missing file run.py or run.ipynb.")


@pytest.mark.parametrize("function_name", FUNCTIONS)
def test_function_exists(function_name: str, student_implementations):
    assert hasattr(
        student_implementations, function_name
    ), f"Missing implemetation for {function_name}."


@pytest.mark.parametrize("function_name", FUNCTIONS)
def test_function_has_docstring(function_name: str, student_implementations):
    fn = getattr(student_implementations, function_name)
    assert fn.__doc__, f"Function {function_name} has no docstring."


def test_black_format(github_repo_path: pathlib.Path):
    python_files = list(github_repo_path.glob("**/*.py"))
    for python_file in python_files:
        content = python_file.read_text()
        try:
            import black
        except ModuleNotFoundError:
            raise ValueError(
                f"We advise you to install the black formater https://github.com/psf/black and format your code with it (not mandatory)."
            )

        try:
            black.format_file_contents(content, fast=True, mode=black.FileMode())
            raise ValueError(
                f"We advise you to format '{python_file.name}' with the black formater https://github.com/psf/black (not mandatory)."
            )
        except black.NothingChanged:
            pass


def test_no_todo_left(github_repo_path: pathlib.Path):
    python_files = list(github_repo_path.glob("**/*.py"))
    for python_file in python_files:
        if python_file.name == pathlib.Path(__file__).name:
            continue  # ignore this file for TODO checks
        content = python_file.read_text()
        assert "todo" not in content.lower(), f"Solve remaining TODOs in {python_file}."


def test_mean_squared_error_gd_0_step(student_implementations, y, tx):
    expected_w = np.array([[0.413044], [0.875757]])
    w, loss = student_implementations.mean_squared_error_gd(y, tx, expected_w, 0, GAMMA)

    expected_w = np.array([[0.413044], [0.875757]])
    expected_loss = 2.959836

    np.testing.assert_allclose(loss, expected_loss, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(w, expected_w, rtol=RTOL, atol=ATOL)
    assert loss.ndim == 0
    assert w.shape == expected_w.shape


def test_mean_squared_error_gd(student_implementations, y, tx, initial_w):
    w, loss = student_implementations.mean_squared_error_gd(
        y, tx, initial_w, MAX_ITERS, GAMMA
    )

    expected_w = np.array([[-0.050586], [0.203718]])
    expected_loss = 0.051534

    np.testing.assert_allclose(loss, expected_loss, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(w, expected_w, rtol=RTOL, atol=ATOL)
    assert loss.ndim == 0
    assert w.shape == expected_w.shape


def test_mean_squared_error_sgd(student_implementations, y, tx, initial_w):
    # n=1 to avoid stochasticity
    w, loss = student_implementations.mean_squared_error_sgd(
        y[:1], tx[:1], initial_w, MAX_ITERS, GAMMA
    )

    expected_loss = 0.844595
    expected_w = np.array([[0.063058], [0.39208]])

    np.testing.assert_allclose(loss, expected_loss, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(w, expected_w, rtol=RTOL, atol=ATOL)
    assert loss.ndim == 0
    assert w.shape == expected_w.shape


def test_least_squares(student_implementations, y, tx):
    w, loss = student_implementations.least_squares(y, tx)

    expected_w = np.array([[0.218786], [-0.053837]])
    expected_loss = 0.026942

    np.testing.assert_allclose(w, expected_w, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(loss, expected_loss, rtol=RTOL, atol=ATOL)
    assert loss.ndim == 0
    assert w.shape == expected_w.shape


def test_ridge_regression_lambda0(student_implementations, y, tx):
    lambda_ = 0.0
    w, loss = student_implementations.ridge_regression(y, tx, lambda_)

    expected_loss = 0.026942
    expected_w = np.array([[0.218786], [-0.053837]])

    np.testing.assert_allclose(w, expected_w, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(loss, expected_loss, rtol=RTOL, atol=ATOL)
    assert loss.ndim == 0
    assert w.shape == expected_w.shape


def test_ridge_regression_lambda1(student_implementations, y, tx):
    lambda_ = 1.0
    w, loss = student_implementations.ridge_regression(y, tx, lambda_)

    expected_loss = 0.03175
    expected_w = np.array([[0.054303], [0.042713]])

    np.testing.assert_allclose(loss, expected_loss, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(w, expected_w, rtol=RTOL, atol=ATOL)
    assert loss.ndim == 0
    assert w.shape == expected_w.shape


def test_logistic_regression_0_step(student_implementations, y, tx):
    expected_w = np.array([[0.463156], [0.939874]])
    y = (y > 0.2) * 1.0
    w, loss = student_implementations.logistic_regression(y, tx, expected_w, 0, GAMMA)

    expected_loss = 1.533694

    np.testing.assert_allclose(loss, expected_loss, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(w, expected_w, rtol=RTOL, atol=ATOL)
    assert loss.ndim == 0
    assert w.shape == expected_w.shape


def test_logistic_regression(student_implementations, y, tx, initial_w):
    y = (y > 0.2) * 1.0
    w, loss = student_implementations.logistic_regression(
        y, tx, initial_w, MAX_ITERS, GAMMA
    )

    expected_loss = 1.348358
    expected_w = np.array([[0.378561], [0.801131]])

    np.testing.assert_allclose(loss, expected_loss, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(w, expected_w, rtol=RTOL, atol=ATOL)
    assert loss.ndim == 0
    assert w.shape == expected_w.shape


def test_reg_logistic_regression(student_implementations, y, tx, initial_w):
    lambda_ = 1.0
    y = (y > 0.2) * 1.0
    w, loss = student_implementations.reg_logistic_regression(
        y, tx, lambda_, initial_w, MAX_ITERS, GAMMA
    )

    expected_loss = 0.972165
    expected_w = np.array([[0.216062], [0.467747]])

    np.testing.assert_allclose(loss, expected_loss, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(w, expected_w, rtol=RTOL, atol=ATOL)
    assert loss.ndim == 0
    assert w.shape == expected_w.shape


def test_reg_logistic_regression_0_step(student_implementations, y, tx):
    lambda_ = 1.0
    expected_w = np.array([[0.409111], [0.843996]])
    y = (y > 0.2) * 1.0
    w, loss = student_implementations.reg_logistic_regression(
        y, tx, lambda_, expected_w, 0, GAMMA
    )

    expected_loss = 1.407327

    np.testing.assert_allclose(loss, expected_loss, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(w, expected_w, rtol=RTOL, atol=ATOL)
    assert loss.ndim == 0
    assert w.shape == expected_w.shape
