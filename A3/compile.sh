#!/bin/bash

module purge
module load compiler/intel/2019u5/intelpython3
export PYTHONPATH=$HOME/.local/lib/python3.6/site-packages:$PYTHONPATH