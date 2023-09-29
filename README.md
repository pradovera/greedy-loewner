# README

This repository contains minimal code for running the numerical examples in the paper:

D. Pradovera, _Toward a certified greedy Loewner framework with minimal sampling_ (2023)

Preprint publicly available [here](https://arxiv.org/abs/2303.01015)!

All examples are from the [SLICOT library](http://slicot.org/20-site/126-benchmark-examples-for-model-reduction)!

## Prerequisites
* **python** (version: 3.8.12)
* **numpy** (version: 1.21.4)
* **scipy** (version: 1.5.3)
* **matplotlib** (version: 3.4.3)

## Execution
The simulations in the paper can be run via `run.py`.

Code can be run as
```
python3 run.py $example_tag
```
The placeholder `$example_tag` can take the self-explanatory values
* `MNA_4`
* `MNA_4_RANDOM`
* `TLINE`
* `TLINE_MEMORY`
* `ISS`
* `ISS_BATCH`
* `ISS_RANDOM`

Otherwise, one can simply run
```
python3 run.py
```
and then input `$example_tag` later.
