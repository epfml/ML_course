# Setup Guide for Coding Machine Learning Methods - EPFL ML Course

In order to implement the algorithms seen in class and work on the projects, we'll be using Python. This first lab will serve as an introduction to the language, the environment we are going to be using, and how to do basic vector and matrix manipulations.

## The environment

### Python distribution: Anaconda
We will be using the [Anaconda](https://www.continuum.io/) distribution to run Python 3, as it is easy to install and comes with all the basic packages we will need. To install Anaconda, go to [the download page](https://www.continuum.io/downloads) and get the Python installer for your OS - make sure to use the newer version 3.x, not 2.x. Follow the instructions of the installer and you're done.
> **Warning!** The installer will ask you if you want to add Anaconda to your path. Your default answer should be yes, unless you have specific reasons not to want this.

### Development Environment 

There are multiple optionsthere. We highly recommend you use [**Jupyter Notebooks**](http://jupyter.org/) which allows for an easy view of what each code section is doing, and is great for exploratory analysis of a dataset. Notebooks are web based, and you start it on your localhost by typing `jupyter notebook` in the console. Notebooks are already available by default by Anaconda. The interface is pretty intuitive, but they are a few tweaks and shortcuts that will make your life easier, which we'll detail in the next section. You can of course ask any of the TAs for help on using the Notebooks.

While you should at least try the Notebooks, if you are more comfortable using a more traditional IDE, you can give [**PyCharm**](https://www.jetbrains.com/pycharm/) a go. Your EPFL email gives you access to the free educational version. You should keep this option in mind if you need a full fledged debugger to find a nasty bug.

And of course, if you want to go raw, you can always use a [decent text editor](https://www.sublimetext.com/) and run your code from the console or any plugin. Keep in mind that the TAs might not be able to help you with your setup if you go down this path.

### The Notebook System

For some ressources on how the notebook system works, you can check

* [The Jupyter notebook beginner's guide](https://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/index.html)
* [The official documentation](http://jupyter-notebook.readthedocs.io/en/latest/index.html)

Note that the Jupyter notebook system is already installed by default by Anaconda, you do not need to install anything else to run it

#### Examples

We provide you with an example of a notebook for [this first lab](https://github.com/epfml/ML_course/tree/master/labs/ex1), but if you want to see some more examples already, feel free to take a look at

* The introductory notebooks available at [Try Jupiter](https://try.jupyter.org/). It spawns an instance of the Jupyter Notebook, which won't save any of your changes.
  *Note: it might not be available if their server is under too much load.*
* [A gallery of interesting IPython Notebooks](https://github.com/ipython/ipython/wiki/A-gallery-of-interesting-IPython-Notebooks) by the Ipython Notebook team
* [Another gallery of notebooks](http://nb.bianp.net/sort/views/)

#### Tips & Tricks

There are a few handy commands that you should start every notebook with


	# Plot figures in the notebook (instead of a new window)
	%matplotlib notebook
	
	# Automatically reload modules
	%load_ext autoreload
	%autoreload 2        
	
	# The usual imports
	import matplotlib.pyplot as plt
	import numpy as np
	import pandas as pd

#### Keyboard shortcuts
* Adding cells
	* `a` adds an empty cell above the selected one,
	* `b` adds it below.
* Running code
	* `Enter` enters the edition mode of the currently selected cell.
	* `Shift-Enter` runs the current cell and goes to the next one.
	* `Ctrl-Enter` runs the currect cell and leave it selected.
* Autocompletion
  * `Tab` pops up the Autocompletion when you are in the middle of writing a function call/class name and shows the arguments of the function being called when used after an opening parenthesis.
  * `Shift-Tab` pops up the help/documentation of the function its used on

* For a complete list of shortcuts, go to `help > keyboard shortcuts`

## Python

We will be working in Python this year. If you already have been introduced to Python, feel free to skip this section. If you come from another background, especially lower level languages, you might want to take some tutorials in addition to this lab in the next week to feel comfortable with it. You do not need to become an expert in Python, but you should be confortable with the general syntax, some of the idiosyncrasies of Python and know how to do basic vector and matrix algebra. For the last part, we will be using NumPy, a library we will introduce later.

For a nice introduction to Python, you should take a look at [the Python tutorial](https://docs.python.org/3/tutorial/index.html). Here are some reading recommendations:

* Skim through Sections 1-3 to get an idea of the Python syntax if you never used it
* Pay a little more attention to Section 4, especially

	* Section 4.2 on for loops, as they behave like `foreach` by default, which may be disturbing if you are more acoutumed to coding in lower level languages.
	* Section 4.7 on functions, default argument values and named arguments, as they are a real pleasure to use (compared to traditional, order-based arguments) once you are used to it.
* Section 5 on Data Structures, especially how to use Lists, Dictionnaries and Tuples if you have not used a language with those concepts before
* You can keep Sections 6-9 on Modules, IO, Exceptions and Objects for later - when you know you will be needing it.
* Section 10 on the standard library and [the standard library index](https://docs.python.org/3/library/index.html) are worth a quick scroll to see what's available.
* Do not bother with Sections 11-16 for now.

Here are some additionnal ressources on Python:

* [Python's standard library reference](https://docs.python.org/3/library/index.html)
* [Debugging and profiling](https://docs.python.org/3/library/debug.html)
* If you want to, some exercises are available at [learnpython.org](http://www.learnpython.org/)


## NumPy and Vector Calculations

Our `npprimer.ipynb` notebook as part of the first lab has some useful commands to help you get started with NumPy.

If you are familiar with Matlab, a good starting point is [this guide](http://docs.scipy.org/doc/numpy-dev/user/numpy-for-matlab-users.html). Be careful that we will use way more the `array` structure compared to the `matrix` structure.

A good and probably more complete reference is [this one](http://www.engr.ucsb.edu/~shell/che210d/numpy.pdf).


### Additional References

[A good Python and NumPy Tutorial from CS231n, Stanford.](http://cs231n.github.io/python-numpy-tutorial/) (_WARNING_ : Uses Python 2 and not 3)
