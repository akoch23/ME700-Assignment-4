# FEA Code Failure Example (Membrane Deflection)

# Library Imports
import os
import gmsh # Import GMSH Python API, necessary for 3D Finite Element Mesh generation for loading into DOLFINx
import dolfinx.io # Import necessary computing environment (FEniCS)
import numpy as np # Import Python Numerical library for specific mathematical operations
import matplotlib.pyplot as plt # Import Python Graphical Plotting library for data plotting, library call abbreviated as "plt"
import ufl # Import Unified Form Language (FEniCS)
import pyvista # Import Python library for 3D data visualization (FE meshs and related animations)
pyvista.start_xvfb()  # Start virtual framebuffer for headless rendering

# Specific DOLFINx imports for FEM operations
from dolfinx.io import gmshio # Tools to convert Gmsh models into DOLFINx mesh structures
from dolfinx.fem.petsc import LinearProblem # High-level interface for linear variational problems
from mpi4py import MPI # Parallel computing via MPI
from dolfinx import fem # FEM functionality (function spaces, BCs, etc.)
from dolfinx import default_scalar_type # Sets float precision for DOLFINx operations
from dolfinx.plot import vtk_mesh # Conversion of DOLFINx meshes to VTK format for PyVista
from dolfinx.io import XDMFFile
from dolfinx import geometry # Geometry tools for queries and evaluations
from pathlib import Path # File and directory path manipulation

# Mesh Generation for Model (2D Circular Disk)
gmsh.initialize()  # Initialize the Gmsh API session
membrane = gmsh.model.occ.addDisk(0, 0, 0, 1, 1) # Create a 2D disk with radius 1 centered at origin
gmsh.model.occ.synchronize() # Finalize CAD operations and synchronize the model
gdim = 2 # Geometric dimension for 2D model
gmsh.model.addPhysicalGroup(gdim, [membrane], 1) # Assign physical group for FEM tagging
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.05) # Set mesh resolution (min)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.05) # Set mesh resolution (max)
gmsh.model.mesh.generate(gdim) # Generate 2D mesh
print("GMSH mesh generated.")

# Convert Gmsh Mesh to DOLFINx Mesh
gmsh_model_rank = 0  # Ensure only one process loads the mesh (MPI parallelism)
mesh_comm = MPI.COMM_WORLD  # MPI communicator for parallelism
domain, cell_markers, facet_markers = gmshio.model_to_mesh(gmsh.model, mesh_comm, gmsh_model_rank, gdim=gdim)
print(f"Rank {MPI.COMM_WORLD.rank}: Mesh conversion to DOLFINx done.")
gmsh.finalize()

# Define Function Space
V = fem.functionspace(domain, ("Lagrange", 1))  # Linear Lagrange function space (scalar field) for displacement
Q = fem.functionspace(domain, ("Lagrange", 5)) # 5th Order Lagrance function space for pressure
print("Function space V defined.")
print("Function space Q defined.")

# Define Mathematical Expression for External Load (Pressure)
x = ufl.SpatialCoordinate(domain)  # Spatial coordinates (symbolic) # Get symbolic spatial coordinate x = (x[0], x[1])
beta = fem.Constant(domain, default_scalar_type(12))  # Controls decay of load
R0 = fem.Constant(domain, default_scalar_type(0.3))  # Offset in y-direction
p = 4 * ufl.exp(-beta**2 * (x[0]**2 + (x[1] - R0)**2))  # Gaussian load centered at y = R0
print("Load expression defined.")

# Define/Apply Boundary Conditions
def on_boundary(x): # Identify boundary points on circle's edge using radius check
    return np.isclose(np.sqrt(x[0]**2 + x[1]**2), 1)  # Dirichlet BC on the circle

boundary_dofs = fem.locate_dofs_geometrical(Q, on_boundary)  # Locate DOFs on boundary (hint: this is the erroneous line)
bc = fem.dirichletbc(default_scalar_type(0), boundary_dofs, V)  # Set u = 0 on boundary (clamped)
print("Boundary conditions applied.")

# Define Variational Problem (Weak Formulation)
u = ufl.TrialFunction(V) # Trial function (unknown displacement)
v = ufl.TestFunction(V) # Test function (virtual displacement)
a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx  # Bilinear form (stiffness matrix)
L = p * v * ufl.dx  # Linear form (load vector)

problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})  # Solver setup, where ksp_type is for direct solver and pc_type is for LU decomposition
print("Problem setup complete. Solving...")
uh = problem.solve()  # Solve for displacement field
print("Solve complete.")

# Interpolate Pressure Field for Visualization
Q = fem.functionspace(domain, ("Lagrange", 5))  # Higher-order space for smoother visualization
expr = fem.Expression(p, Q.element.interpolation_points())  # Interpolation of p to Q
pressure = fem.Function(Q)
pressure.interpolate(expr)
print("Pressure interpolated.")


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


# Extract and Plot 2D Data (y-axis)
tol = 0.001
y = np.linspace(-1 + tol, 1 - tol, 101)  # Avoid edges to stay inside mesh
points = np.zeros((3, 101))
points[1] = y  # Points lie along vertical line x=0
u_values = []
p_values = []

# Bounding-box tree for fast point-in-cell queries
bb_tree = geometry.bb_tree(domain, domain.topology.dim)

# Find mesh cells containing the line points

cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)

points_on_proc = []
cells = []
for i, point in enumerate(points.T):
    if len(colliding_cells.links(i)) > 0:
        points_on_proc.append(point)
        cells.append(colliding_cells.links(i)[0])

points_on_proc = np.array(points_on_proc, dtype=np.float64)
u_values = uh.eval(points_on_proc, cells)  # Evaluate displacement
p_values = pressure.eval(points_on_proc, cells)  # Evaluate pressure
if len(points_on_proc) == 0:
    print(f"Rank {MPI.COMM_WORLD.rank}: No points found.")

# Plot 1D Profiles of Displacement and Pressure

fig = plt.figure()
plt.plot(points_on_proc[:, 1], 50 * u_values, "k", linewidth=2, label="Deflection ($\\times 50$)")
plt.plot(points_on_proc[:, 1], p_values, "b--", linewidth=2, label="Load")
plt.grid(True)
plt.xlabel("y")
plt.legend()
# If run in parallel as a python file, save a plot per processor
plt.savefig(f"membrane_rank{MPI.COMM_WORLD.rank:d}.png")
