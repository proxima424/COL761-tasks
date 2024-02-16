#!/bin/bash
git clone https://github.com/proxima424/COL761-tasks.git

folder="COL761-tasks"
cd "$folder"

module load compiler/python/3.6.0/ucs4/gnu/447
module load pythonpackages/3.6.0/matplotlib/3.0.2/gnu
module load pythonpackages/3.6.0/numpy/1.16.1/gnu
module load pythonpackages/3.6.0/pandas/0.23.4/gnu
module load pythonpackages/3.6.0/scikit-learn/0.21.2/gnu
module load pythonpackages/3.6.0/scipy/1.1.0/gnu
