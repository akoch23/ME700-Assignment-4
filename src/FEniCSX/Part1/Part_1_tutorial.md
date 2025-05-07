# Tutorial for Part 1 Code (3D Membrane Deflection Problem)

This tutorial will cover the function and expect output of Part_1_Example.py within this repository, based on the implementation guide of similar name from the official FEniCSx tutorial handbook.

## Creating the Mesh
To create the digital geometry of the desired shape, the Python-API of GMSH is used. 

``` python
# Library Imports
import gmsh # Import GMSH Python API, necessary for 3D Finite Element Mesh generation for loading into DOLFINx

# Mesh Generation for Model (2D Circular Disk)
gmsh.initialize()  # Initialize the Gmsh API session
membrane = gmsh.model.occ.addDisk(0, 0, 0, 1, 1) # Create a 2D disk with radius 1 centered at origin
gmsh.model.occ.synchronize() # Finalize CAD operations and synchronize the model
gdim = 2 # Geometric dimension for 2D model
gmsh.model.addPhysicalGroup(gdim, [membrane], 1) # Assign physical group for FEM tagging
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.05) # Set mesh resolution (min)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.05) # Set mesh resolution (max)
gmsh.model.mesh.generate(gdim) # Generate Final 2D mesh
print("GMSH mesh generated.")
```

After this particular section of code, you should recieve this information at output (you can set a breakpoint after the last above line to test this):
```
Info    : Meshing 1D...
Info    : Meshing curve 1 (Ellipse)
Info    : Done meshing 1D (Wall 0.00019836s, CPU 0.00016s)
Info    : Meshing 2D...
Info    : Meshing surface 1 (Plane, Frontal-Delaunay)
Info    : Done meshing 2D (Wall 0.0740319s, CPU 0.073866s)
Info    : 1550 nodes 3099 elements
GMSH mesh generated.
```

## Importing Generated Mesh (GMSH to DOLFINx)

```python
# Specific Library Imports for DOLFINx Operations
from dolfinx.io import gmshio # Tools to convert Gmsh models into DOLFINx mesh structures
from dolfinx.fem.petsc import LinearProblem # High-level interface for linear variational problems
from mpi4py import MPI # Parallel computing via MPI

# Convert Gmsh Mesh to DOLFINx Mesh
gmsh_model_rank = 0  # Ensure only one process loads the mesh (MPI parallelism)
mesh_comm = MPI.COMM_WORLD  # MPI communicator for parallelism
domain, cell_markers, facet_markers = gmshio.model_to_mesh(gmsh.model, mesh_comm, gmsh_model_rank, gdim=gdim)
print(f"Rank {MPI.COMM_WORLD.rank}: Mesh conversion to DOLFINx done.")
gmsh.finalize()

# Define Function Space
V = fem.functionspace(domain, ("Lagrange", 1))  # Linear Lagrange function space (scalar field) for displacement
print("Function space V defined.")
```
