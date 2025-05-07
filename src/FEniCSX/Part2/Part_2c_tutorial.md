# Part C: Failure of FEA Method (Membrane Deflection)

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

Say I wanted to define the function space for both pressure and displacement right away/in the same section instead of right before their respective use cases. (See # Define Function Space code block)

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
Q = fem.functionspace(domain, ("Lagrange", 5)) # 5th Order Lagrange function space for pressure
print("Function space V defined.")
print("Function space Q defined.")
```

## Define Spatially Varying Load

``` python
# Define Mathematical Expression for External Load (Pressure)
x = ufl.SpatialCoordinate(domain)  # Spatial coordinates (symbolic) # Get symbolic spatial coordinate x = (x[0], x[1])
beta = fem.Constant(domain, default_scalar_type(12))  # Controls decay of load
R0 = fem.Constant(domain, default_scalar_type(0.3))  # Offset in y-direction
p = 4 * ufl.exp(-beta**2 * (x[0]**2 + (x[1] - R0)**2))  # Gaussian load centered at y = R0
print("Load expression defined.")
```

## Establish Boundary Condition (Dirichlet)

Say I also happen to accidentally assign different function spaces to the locate_dofs_geometrical function and dirichletbc function.

```python
# Define/Apply Boundary Conditions
def on_boundary(x): # Identify boundary points on circle's edge using radius check
    return np.isclose(np.sqrt(x[0]**2 + x[1]**2), 1)  # Dirichlet BC on the circle

boundary_dofs = fem.locate_dofs_geometrical(Q, on_boundary)  # Locate DOFs on boundary (hint: this is the erroneous line)
bc = fem.dirichletbc(default_scalar_type(0), boundary_dofs, V)  # Set u = 0 on boundary (clamped)
print("Boundary conditions applied.")
```

This leads to this output:

```
Info    : Meshing 1D...
Info    : Meshing curve 1 (Ellipse)
Info    : Done meshing 1D (Wall 0.000174386s, CPU 0.00014s)
Info    : Meshing 2D...
Info    : Meshing surface 1 (Plane, Frontal-Delaunay)
Info    : Done meshing 2D (Wall 0.0724326s, CPU 0.072132s)
Info    : 1550 nodes 3099 elements
GMSH mesh generated.
Rank 0: Mesh conversion to DOLFINx done.
Function space V defined.
Function space Q defined.
Load expression defined.
Boundary conditions applied.
Problem setup complete. Solving...
Segmentation fault
```

These two functions must use the same function space, as otherwise the boundary_dofs will be located using the geometry of one function space, but apply them as if they belong to the other functions space, leading to undefined behavior and consequently a segmentation fault during the process of applying the function over the domain and altering the created mesh.
