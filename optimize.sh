#!/bin/bash

python3 tests/test_optimize_mp.py config config/fantasia_$1_$2_ga.json && python3 tests/plot_result.py $1 $2 && cp -r figs ~/public_html/
