# nemati-collab

**Note:** This repository is from 2017 when I worked in the Molkov Lab, Neuroscience Institute, Georgia State University; I updated it recently (November, 2026) to make sure the code still runs. Obligatory AI disclaimer: I used Claude Code when performing the recent updates--- limited to aesthetic, ease-of-use.

## Summary & Objectives

### Objectives

Elucidate the cardiorespiratory circuitry underlying autonomic control of breathing, blood pressure, heart rate. This repository includes code related to analysis and model-fitting. The sample configurations are for the sample dataset I used (PhysioNet, Fantasia ECG dataset).

### Code Summary

The repository contains the latest version of the cardiovascular model (implemented in C) and optimized data processing code written in C. The model optimization module (fit_model) is accessible from Python.

## Example scripts

#### Produce model output file (.npy format) for subject 'f1y02':
```bash
$ python3 tests/test_model.py f1y02
```

#### Produce a CVC synchrogram plot:
```bash
$ python3 phase_time_plot.py
```

## Installation

### Dependencies

- Python >=3.5,<3.12 (developed with 3.6.5)
- wfdb-python == 1.3.7

#### [extra] Required for model optimization

- GNU Scientific Library (GSL) >= 2.0 (developed with 2.4)

### Install steps

First, install all the dependencies.

1. Change the "LIB_PATH" variable in the config file that you want (./config/*.json) so that it points to the git repository. See the config/ directory for examples.

2. Change the "path" variable in the config file so that it points to the location of the dataset. See the config/ directory for examples.

3. Set the "database" variable in the config file (probably 'fantasia').

4. Set any other variables in the config file so that they do what you want. See the examples below for an idea of what variables you might change.

5. Build the C sources by running the build script. Check for errors.

6. Install the python package using `uv`.

```bash

$ uv sync

```

To get started, run the scripts as in the examples below.

## Example usage

### Test the model
```bash

rocapp@ubuntu-server:~/Git/nemati-collab$ python3 tests/test_model.py

```

### Print the help message for the optimization test
```bash

rocapp@ubuntu-server:~/Git/nemati-collab$ python3 tests/test_optimize_mp.py help

Usage:

  python3 tests/test_optimize_mp.py [args]

Args:

  subject_name : f1y02
  opt_feature : rsa
  method : basinhopping
  tol : 1e-05
  simtime : 60
  nr_p : 2
  T : 0.5
  stepsize : 1e-06
  niter : 10000
  niter_success : 1500
  interval : 10


```

### Run the optimization test
```bash

rocapp@ubuntu-server:~/Git/nemati-collab$ python3 tests/test_optimize_mp.py config config/fantasia_f1y02_rsa.json

```

### Load statistics for all subjects
```python

import pycardio as pycard
stats = pycard.load_data.load_all_stats(database='fantasia', path='/home/rocapp/Public')
averages = pycard.process_data.get_averages(stats)

```

## Package structure

### The package is structured like so:

```bash
nemati-collab/          ||      "Top-level repository directory"
├── scripts             ||      "Stand-alone scripts (these are for development use only)"
├── pycardio            ||      "The pycardio package source directory"
│   ├── c_fit-model     ||      "C source files for optimization and sampling procedures (e.g. CMAES)"
│   ├── c_model         ||      "C source files for integrating the ODE-based cardiovascular model"
│   ├── c_process-data  ||      "C implementation of data processing functions"
│   ├── fit_model       ||      "Python module for fitting the model parameters to physiological data"
│   ├── load_data       ||      "Python module for loading physiological data"
│   ├── model           ||      "Python wrapper to instantiate the model (accesses c_model)"
│   └── process_data    ||      "Python data processing module (accesses c_process-data)"
└── tests               ||      "Tests of various features"
```
