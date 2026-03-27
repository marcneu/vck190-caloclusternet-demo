# VCK190 CaloClusterNet Demonstrator

## Preliminaries

Make sure the following packages are installed on your system

```bash
python3.12
opencv2
```

You need the following Python Packages

```bash
setuptools
flask
numpy
```

## Install

Either use 

```bash
pip install -e .
```

or 

```bash
python setup.py bdist_wheel --plat-name linux_aarch64
```

to build a distributable wheel

## How to run the standalone inference

```bash
python standalone.py binaries/accelerator.xclbin data/input.h5 output.h5 --num-events 1024
```

## How To run the server

```bash
python3 main.py
```

