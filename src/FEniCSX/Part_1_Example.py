import gmsh # Import GMSH Python API, necessary for 3D Finite Element Mesh generation for loading into DOLFINx
import dolfinx.io # Import necessary computing environment (FEniCS)
import numpy as np # Import Python Numerical library for specific mathematical operations
import matplotlib.pyplot as plt # Import Python Graphical Plotting library for data plotting, library call abbreviated as "plt"
import ufl # Import Unified Form Language (FEniCS)
import pyvista # Import Python library for 3D data visualization (FE meshs and related animations)

# Specific Imports
from dolfinx.io import gmshio  # Convert GMSH models into DOLFINx-compatible meshes
from dolfinx.fem.petsc import LinearProblem  # Simplified linear solver interface
from mpi4py import MPI  # MPI support for parallel execution
from dolfinx import fem  # Finite element tools (Function, BCs, FunctionSpace)
from dolfinx import default_scalar_type  # Default float type (float64 or float32)
from dolfinx.plot import vtk_mesh  # Convert mesh data for VTK-based tools
from dolfinx import geometry  # Tools for spatial geometry queries
from pathlib import Path  # Filesystem path manipulation

# Mesh Generation for Model (2D Circular Disk)
gmsh.initialize()  # Start GMSH session
membrane = gmsh.model.occ.addDisk(0, 0, 0, 1, 1)  # Add a 2D circular disk (unit radius)
gmsh.model.occ.synchronize()  # Finalize geometry changes
gdim = 2  # Geometric dimension (2D)
gmsh.model.addPhysicalGroup(gdim, [membrane], 1)  # Tag the disk for FEM reference
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.05)  # Mesh size control (min)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.05)  # Mesh size control (max)
gmsh.model.mesh.generate(gdim)  # Generate 2D mesh

# Convert Mesh to FEniCSx
gmsh_model_rank = 0  # Only this rank loads the mesh (others will get it via MPI)
mesh_comm = MPI.COMM_WORLD  # MPI communicator for parallelism
domain, cell_markers, facet_markers = gmshio.model_to_mesh(gmsh.model, mesh_comm, gmsh_model_rank, gdim=gdim)

# Define Function Space
V = fem.functionspace(domain, ("Lagrange", 1))  # Linear Lagrange function space for displacement

# Define Mathematical Expression for Load (Pressure)
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
