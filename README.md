# ME700 Assignment 4

[![python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
![os](https://img.shields.io/badge/os-ubuntu%20|%20macos%20|%20windows-blue.svg)

## Part 1

### Conda environment installation and testing

To install this package, first establish a new conda environment:
```bash
conda create -n fenicsx-env python=3.12
```
Afterwards, activate the environment:
```bash
conda activate fenicsx-env
```

You can double check if the installed version of python is indeed version 3.12 in the new conda environment:
```bash
python --version
```

Ensure that pip is using the latest version of setuptools:
```bash
pip install --upgrade pip setuptools wheel
```

Follow by installing DolfinX and other necessary libraries:
```bash
conda install -c conda-forge fenics-dolfinx mpich pyvista
```

Finally, install remaining libraries:
```bash
pip install imageio
pip install gmsh
```
