import doctest
import io
import sys
import numpy as np

"""
This is a helper function that you can use to add simple unit tests
to your exercise.

This uses https://docs.python.org/3/library/doctest.html.
"""


def test(f):
    """Run unit tests defined in a function's docstring (doctests)"""
    tests = doctest.DocTestFinder().find(f)
    assert len(tests) <= 1
    for test in tests:
        # We redirect stdout to a string, so we can tell if the tests worked out or not
        orig_stdout = sys.stdout
        sys.stdout = io.StringIO()

        orig_rng_state = np.random.get_state()

        try:
            np.random.seed(1)
            results: doctest.TestResults = doctest.DocTestRunner().run(test)
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = orig_stdout
            np.random.set_state(orig_rng_state)

        if results.failed > 0:
            print(f"❌ The are some issues with your implementation of `{f.__name__}`:")
            print(output, end="")
            print(
                "**********************************************************************"
            )
        elif results.attempted > 0:
            print(f"✅ Your `{f.__name__}` passes some basic tests.")
        else:
            print(f"Could not find any tests for {f.__name__}")
