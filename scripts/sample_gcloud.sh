#!/bin/bash
time python3.6 tests/test_sampler_gcloud.py $1 $2 && python3.6 tests/plot_sampler.py $1 $2 && cp -r figs ~/public_html/
