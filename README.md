# ME700 Assignment 4

[![python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
![os](https://img.shields.io/badge/os-ubuntu%20|%20macos%20|%20windows-blue.svg)

## Conda environment installation and testing
### Step 1
To install this package, first establish a new conda environment:
```bash
conda create -n fenicsx-env python=3.12
```
### Step 2
Afterwards, activate the environment:
```bash
conda activate fenicsx-env
```
### Step 2.1
You can double check if the installed version of python is indeed version 3.12 in the new conda environment:
```bash
python --version
```
### Step 2.2
Ensure that pip is using the latest version of setuptools:
```bash
pip install --upgrade pip setuptools wheel
```
### Step 3
Follow by installing DolfinX and other necessary libraries:
Note for Windows users: The below command, which is listed on the Fenicsx official download site, will work for the installion of fenics-dolfinx and pyvista, but NOT mpich. The petsc library will also not install properly on Windows. This can be circumvented by using Docker or WSL2 (Virtual Linux Interface).
```bash
conda install -c conda-forge fenics-dolfinx mpich pyvista
```

### Step 4 (Optional, important if using personal VS code editor)
Finally, install remaining libraries:
```bash
pip install imageio
pip install gmsh
pip install pyvista
pip install petsc
pip install mpi4py
pip install ipykernel (if planning to use jupyter)
```
Note: The lastest version of pyvista and vtk may not be compatible with each other, so make sure to look up which versions will work for you.

Alternatively, use the SCC VS code server for full access to said libraries, but still follow through Steps 1 - 3. In addition, you may use the mamba command instead of the conda command for faster results.

## Important: For all assignment parts, please refer to the main (src) folder and the Part 1 and 2 folders as needed (Part 2 folder contains Assignment 4 Parts A, B, and C).

### Part A:

This script uses a finite element method (FEM via DolfinX library) with parallel processing (mpi4py) in Python to solve a Poisson equation numerically on a unit square domain, then validates and visualizes the solution in both 2D and 3D. The script sets up a finite element function space using Lagrange polynomials of degree 1 for the unknown solution and defines the weak formulation of the Poisson equation. The linear system is solved using the PETSc solver with preconditioned LU decomposition. After solving, the script calculates the L2 error between the numerical solution and the exact solution in a higher-order function space (degree 2), and computes the maximum pointwise error. The results are printed on the root processor. Lastly, PyVista is used to plot the mesh and the computed solution. The mesh is first displayed in 2D, then the solution is visualized as a surface plot, and finally, the mesh is warped in 3D based on the solution values for enhanced visualization of the effect of the load.

### Part B:

This script performs a mesh refinement study to simulate/solve for the deflection of a beam structure under load (the effect of mesh refinement: h-refinement) and polynomial degree (p-refinement) on the computed maximum displacement in a clamped 3D beam subject to gravitational loading using the DOLFINx FEM library. The beam is modeled as a rectangular prism with a length of 1 unit and a square cross-section of width and height 0.2 units. It is clamped at one end (x = 0) and subjected to a gravitational body force in the negative z-direction. 
The h-refinement is demonstrated by increasing the number of hexahedral elements along the beam’s length from 2 to 24, and p-refinement by comparing linear (first-order) and quadratic (second-order) Lagrange elements. For each combination of mesh resolution and element order, the code solves the static linear elasticity problem using LU decomposition and computes the maximum displacement experienced by the beam. These values are recorded and visualized in a plot (Mesh Refinement.png) that shows the relationship between mesh density and maximum displacement for both element orders and eventual solution convergence.

### Part C:

This script simulates and visualizes the deflection of a circular elastic membrane subjected to a localized pressure load using the finite element method (FEM via DolfinX library), with results visualized via PyVista. However, the boundary conditions are incorrectly applied (wrong function space used for DOF locating/assembly) which causes the simulation to fail because the problem cannot properly assembly DOF for one function space with the DOFs of another. The finite element solver cannot properly properly assign system boundaries or determine how the solution should behave at said boundaries, leading to a segmentation fault within dolfinx.
