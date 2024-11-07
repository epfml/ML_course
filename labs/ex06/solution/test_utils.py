import doctest
import io
import sys
import numpy as np


def test(f):
    # The `globs` defines the variables, functions and packages allowed in the docstring.
    tests = doctest.DocTestFinder().find(f)
    assert len(tests) <= 1
    for test in tests:
        # We redirect stdout to a string, so we can tell if the tests worked out or not
        orig_stdout = sys.stdout
        sys.stdout = io.StringIO()

        try:
            print("Hey!")
            results: doctest.TestResults = doctest.DocTestRunner().run(test)
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = orig_stdout

        if results.failed > 0:
            print(f"❌ The are some issues with your implementation of `{f.__name__}`:")
            print(output, end="")
            print(
                "**********************************************************************"
            )
        elif results.attempted > 0:
            print(f"✅ Your `{f.__name__}` passed {results.attempted} tests.")
        else:
            print(f"Could not find any tests for {f.__name__}")
