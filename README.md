# ME700 Assignment 4

[![python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
![os](https://img.shields.io/badge/os-ubuntu%20|%20macos%20|%20windows-blue.svg)

## Conda environment installation and testing

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

Alternatively, you may use the mamba command for faster results. In either case, installation of libraries will likely take several minutes, so do not be alarmed if it appears to be taking an unusually long time.

## Important: For all assignment parts, please refer to the main (src) folder and the Part 1 and 2 folders as needed (Part 2 folder contains Assignment 4 Parts A, B, and C).

### Part A:

This script uses a finite element method (FEM via DolfinX library) with parallel processing (mpi4py) in Python to solve a Poisson equation numerically on a unit square domain, then validates and visualizes the solution in both 2D and 3D. The script sets up a finite element function space using Lagrange polynomials of degree 1 for the unknown solution and defines the weak formulation of the Poisson equation. The linear system is solved using the PETSc solver with preconditioned LU decomposition. After solving, the script calculates the L2 error between the numerical solution and the exact solution in a higher-order function space (degree 2), and computes the maximum pointwise error. The results are printed on the root processor. Lastly, PyVista is used to plot the mesh and the computed solution. The mesh is first displayed in 2D, then the solution is visualized as a surface plot, and finally, the mesh is warped in 3D based on the solution values for enhanced visualization.

### Part B:

This script performs a mesh refinement study to simulate/solve for the deflection of a disk-shaped membrane under a pressure load using the finite element method (FEM via DolfinX library). The script leverages DolfinX for solving the governing partial differential equation, and GMSH for mesh generation. The study uses a coarse mesh as a base, and proceeds to perform mesh refinement via h-refinement (reducing mesh element size and consequently also reducing overall length) and p-refinement (higher polynomial order for the finite element function space).
The run_simulation function solves the linearized membrane problem using Dirichlet boundary conditions (fixed boundary) on the circular domain. The pressure distribution p(x, y) is given by an exponential function, and the problem is discretized using Lagrange polynomial elements of degree 1. The solution is obtained by solving a linear problem, with the boundary conditions and pressure term incorporated into the weak form. After solving the problem for each case, the script computes the deflection and pressure values at a series of points along the boundary of the membrane using the eval() function to evaluate the solution at specified coordinates. The results are then plotted, displaying deflection of the membrane (scaled by a factor of 50) and the applied pressure, with the plots saved as PNG images for each simulation setup.

### Part C:

Using similar code to Part B, this script also attempts to solve a membrane deflection problem using the finite element method (FEM via DolfinX library), with results visualized via PyVista. However, this time, the boundary condition is intended to enforce zero displacement is removed, causing the simulation to fail because the problem lacks essential boundary conditions. Without them, the finite element solver has no information about how the solution should behave at the boundary. Without correctly application the necessary constraints to the solution at the edges of the membrane, the solver cannot find a valid solution.
