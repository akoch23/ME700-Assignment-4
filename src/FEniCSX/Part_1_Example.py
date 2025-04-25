# Library Imports
import gmsh # Import GMSH Python API, necessary for 3D Finite Element Mesh generation for loading into DOLFINx
import dolfinx.io # Import necessary computing environment (FEniCS)
import numpy as np # Import Python Numerical library for specific mathematical operations
import matplotlib.pyplot as plt # Import Python Graphical Plotting library for data plotting, library call abbreviated as "plt"
import ufl # Import Unified Form Language (FEniCS)
import pyvista # Import Python library for 3D data visualization (FE meshs and related animations)

# Specific DOLFINx imports for FEM operations
from dolfinx.io import gmshio  # Tools to convert Gmsh models into DOLFINx mesh structures
from dolfinx.fem.petsc import LinearProblem  # High-level interface for linear variational problems
from mpi4py import MPI  # Parallel computing via MPI
from dolfinx import fem  # FEM functionality (function spaces, BCs, etc.)
from dolfinx import default_scalar_type  # Sets float precision for DOLFINx operations
from dolfinx.plot import vtk_mesh  # Conversion of DOLFINx meshes to VTK format for PyVista
from dolfinx import geometry  # Geometry tools for queries and evaluations
from pathlib import Path  # File and directory path manipulation

# Mesh Generation for Model (2D Circular Disk)
gmsh.initialize()  # Initialize the Gmsh API session
membrane = gmsh.model.occ.addDisk(0, 0, 0, 1, 1)  # Create a 2D disk with radius 1 centered at origin
gmsh.model.occ.synchronize()  # Finalize CAD operations and synchronize the model
gdim = 2  # Geometric dimension for 2D model
gmsh.model.addPhysicalGroup(gdim, [membrane], 1)  # Assign physical group for FEM tagging
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.05)  # Set mesh resolution (min)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.05)  # Set mesh resolution (max)
gmsh.model.mesh.generate(gdim)  # Generate 2D mesh

Convert Gmsh Mesh to DOLFINx Mesh
gmsh_model_rank = 0  # Ensure only one process loads the mesh (MPI parallelism)
mesh_comm = MPI.COMM_WORLD  # MPI communicator for parallelism
domain, cell_markers, facet_markers = gmshio.model_to_mesh(gmsh.model, mesh_comm, gmsh_model_rank, gdim=gdim)

# Define Function Space
V = fem.functionspace(domain, ("Lagrange", 1))  # Linear Lagrange function space (scalar field) for displacement

# Define Mathematical Expression for External Load (Pressure)
x = ufl.SpatialCoordinate(domain)  # Spatial coordinates (symbolic) 
beta = fem.Constant(domain, default_scalar_type(12))  # Controls decay of load
R0 = fem.Constant(domain, default_scalar_type(0.3))  # Offset in y
p = 4 * ufl.exp(-beta**2 * (x[0]**2 + (x[1] - R0)**2))  # Gaussian load centered at y = R0

# Define Boundary Conditions
def on_boundary(x):
    return np.isclose(np.sqrt(x[0]**2 + x[1]**2), 1)  # Dirichlet BC on the circle

boundary_dofs = fem.locate_dofs_geometrical(V, on_boundary)  # Find DOFs on boundary
bc = fem.dirichletbc(default_scalar_type(0), boundary_dofs, V)  # Set u = 0 on boundary

# Define Variational Problem
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx  # Bilinear form (stiffness matrix)
L = p * v * ufl.dx  # Linear form (load vector)

problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})  # Solver setup
uh = problem.solve()  # Solve for displacement

# Interpolate Pressure Field for Visualization
Q = fem.functionspace(domain, ("Lagrange", 5))  # Higher-order space for smoother visualization
expr = fem.Expression(p, Q.element.interpolation_points())  # Interpolation of p to Q
pressure = fem.Function(Q)
pressure.interpolate(expr)

# PyVista Visualization
pyvista.start_xvfb()  # Start virtual framebuffer for headless rendering

# Extract topology from mesh and create pyvista mesh
topology, cell_types, x = vtk_mesh(V)  # Extract mesh for VTK
grid = pyvista.UnstructuredGrid(topology, cell_types, x)  # PyVista grid from mesh

# Set deflection values and add it to plotter
grid.point_data["u"] = uh.x.array  # Attach displacement data
warped = grid.warp_by_scalar("u", factor=25)  # Exaggerate displacement for viz

plotter = pyvista.Plotter()
plotter.add_mesh(warped, show_edges=True, show_scalar_bar=True, scalars="u")
if not pyvista.OFF_SCREEN:
    plotter.show()
else:
    plotter.screenshot("deflection.png")

# Visualize Pressure Field
load_plotter = pyvista.Plotter()
p_grid = pyvista.UnstructuredGrid(*vtk_mesh(Q))  # Create PyVista grid for pressure
p_grid.point_data["p"] = pressure.x.array.real  # Attach pressure data
warped_p = p_grid.warp_by_scalar("p", factor=0.5)  # Warp to visualize pressure
warped_p.set_active_scalars("p")
load_plotter.add_mesh(warped_p, show_scalar_bar=True)
load_plotter.view_xy()
if not pyvista.OFF_SCREEN:
    load_plotter.show()
else:
    load_plotter.screenshot("load.png")

# Extract and Plot 1D Data Along Vertical Line (y-axis)
tol = 0.001
y = np.linspace(-1 + tol, 1 - tol, 101)  # Avoid edges to stay inside mesh
points = np.zeros((3, 101))
points[1] = y  # Points lie along vertical line x=0

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

# Plot Vertical Profiles
fig = plt.figure()
plt.plot(points_on_proc[:, 1], 50 * u_values, "k", linewidth=2, label="Deflection ($\\times 50$)")
plt.plot(points_on_proc[:, 1], p_values, "b--", linewidth=2, label="Load")
plt.grid(True)
plt.xlabel("y")
plt.legend()
# If run in parallel as a python file, save a plot per processor
plt.savefig(f"membrane_rank{MPI.COMM_WORLD.rank:d}.png")

# Write Results to Disk
pressure.name = "Load"
uh.name = "Deflection"
results_folder = Path("results")
results_folder.mkdir(exist_ok=True, parents=True)

# Write pressure and deflection fields in VTX format
with dolfinx.io.VTXWriter(MPI.COMM_WORLD, results_folder / "membrane_pressure.bp", [pressure], engine="BP4") as vtx:
    vtx.write(0.0)
with dolfinx.io.VTXWriter(MPI.COMM_WORLD, results_folder / "membrane_deflection.bp", [uh], engine="BP4") as vtx:
    vtx.write(0.0)
