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

```python
# Define/Apply Boundary Conditions
def on_boundary(x): # Identify boundary points on circle's edge using radius check
    return np.isclose(np.sqrt(x[0]**2 + x[1]**2), 1)  # Dirichlet BC on the circle

boundary_dofs = fem.locate_dofs_geometrical(V, on_boundary)  # Locate DOFs on boundary
bc = fem.dirichletbc(default_scalar_type(0), boundary_dofs, V)  # Set u = 0 on boundary (clamped)
print("Boundary conditions applied.")
```

## Define Variational Problem (Weak Formulation)

```python
# Define Variational Problem (Weak Formulation)
u = ufl.TrialFunction(V) # Trial function (unknown displacement)
v = ufl.TestFunction(V) # Test function (virtual displacement)
a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx  # Bilinear form (stiffness matrix)
L = p * v * ufl.dx  # Linear form (load vector)

problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})  # Solver setup, where ksp_type is for direct solver and pc_type is for LU decomposition
print("Problem setup complete. Solving...")
uh = problem.solve()  # Solve for displacement field
print("Solve complete.")
```

## Interpolation of Spatiality Varying Load Function into Function Space (Visualization)

```python
# Interpolate Pressure Field for Visualization
Q = fem.functionspace(domain, ("Lagrange", 5))  # Higher-order space for smoother visualization
expr = fem.Expression(p, Q.element.interpolation_points())  # Interpolation of p to Q
pressure = fem.Function(Q)
pressure.interpolate(expr)
print("Pressure interpolated.")
```

## Plotting Deflection (u_h) over Domain

```python
# PyVista Visualization of Deformed Mesh
print("Starting PyVista visualization setup...")

# Extract topology from mesh and create displacement pyvista mesh
topology, cell_types, x = vtk_mesh(uh.function_space)  # Extract mesh for VTK
grid = pyvista.UnstructuredGrid(topology, cell_types, x.copy())  # PyVista grid from mesh

# Ensure the warp is applied to z-direction only by rotating mesh
grid.points[:, 2] = 0.0  # flatten to XY plane if necessary
# Set deflection values and add it to plotter
grid.point_data["u"] = uh.x.array  # Attach displacement data
warped = grid.warp_by_scalar("u", factor=25) # Exaggerate displacement for visualization of displacement
print("PyVista grid for displacement created.")

print("Displacement shape:", uh.x.array.shape)
print("Number of points in mesh:", grid.points.shape)

if uh.x.array.shape[0] != grid.points.shape[0]:
    print("Mismatch between scalar data and mesh points. Skipping warp.")
else:
    grid.point_data["u"] = uh.x.array
    warped = grid.warp_by_scalar("u", factor=25)
    print("Warping successful.")

"""
plotter = pyvista.Plotter()
plotter.open_gif("deformed_membrane.gif")
plotter.add_mesh(warped, show_edges=True, show_scalar_bar=True, scalars="u")
plotter.write_frame()
plotter.close()
"""

plotter = pyvista.Plotter(off_screen=True) # Off screen argument necessary
plotter.add_mesh(warped, show_edges=True, show_scalar_bar=True, scalars="u")
plotter.view_xy()
plotter.camera.zoom(2.0)
plotter.camera_position = [
    (5, 5, 5),   # camera location
    (0, 0, 0),      # focal point (where the camera looks)
    (0, 0, 1),      # up direction
]

# This triggers rendering and saves an image
plotter.render()
plotter.screenshot("deformed_membrane.png")
plotter.close()
print("Deformation image generated.")
```

## Plot Load over Domain

```python
# PyVista Visualization of Pressure Field
p_grid = pyvista.UnstructuredGrid(*vtk_mesh(Q))  # Create PyVista grid for pressure
p_grid.point_data["p"] = pressure.x.array.real  # Attach pressure data
warped_p = p_grid.warp_by_scalar("p", factor=0.5)  # Warp to visualize pressure
warped_p.set_active_scalars("p")
print("PyVista grid for pressure field created.")

print("Pressure field shape:", pressure.x.array.real)
# print("Number of points in mesh:")


plotter = pyvista.Plotter(off_screen=True)
plotter.add_mesh(warped_p, show_scalar_bar=True)
plotter.view_xy()
plotter.camera.zoom(1.25)
plotter.camera_position = [
    (5, 5, 5),   # camera location
    (0, 0, 0),      # focal point (where the camera looks)
    (0, 0, 1),      # up direction
]

plotter.render()
plotter.screenshot("pressure_field.png")
plotter.close()
print("Pressure plot image generated.")
```
