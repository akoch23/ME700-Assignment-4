# Mesh Refinement Study

import gmsh
gmsh.initialize()

# Mesh Generation (gmsh)
membrane = gmsh.model.occ.addDisk(0, 0, 0, 1, 1) # Create disk-shaped mesh of radius 1
gmsh.model.occ.synchronize() # Finalizes the CAD model before meshing
gdim = 2
gmsh.model.addPhysicalGroup(gdim, [membrane], 1) # Label the domain (disk) as a physical region, necessary for DolfinX conversion

gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.05) # Controls minimum length
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.05) # Controls maximum length 
gmsh.model.mesh.generate(gdim) # Generates 2D mesh

# Convert gmesh construct to usable DolfinX structure
from dolfinx.io import gmshio
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI

gmsh_model_rank = 0
mesh_comm = MPI.COMM_WORLD
domain, cell_markers, facet_markers = gmshio.model_to_mesh(gmsh.model, mesh_comm, gmsh_model_rank, gdim=gdim)

# Function Space and Load Definition
from dolfinx import fem
V = fem.functionspace(domain, ("Lagrange", 1)) # Defines the function space for deflection using first-order Lagrange (linear) elements

import ufl
from dolfinx import default_scalar_type
x = ufl.SpatialCoordinate(domain)
beta = fem.Constant(domain, default_scalar_type(12))
R0 = fem.Constant(domain, default_scalar_type(0.3)) 
p = 4 * ufl.exp(-beta**2 * (x[0]**2 + (x[1] - R0)**2)) # Defines concentrated Gaussian load function, which peaks around 0.3

import numpy as np

# Boundary Condition Definition
def on_boundary(x):
    return np.isclose(np.sqrt(x[0]**2 + x[1]**2), 1) # Detects if any point is on the circular boundary of the mesh by substituing values into equation of a circle

boundary_dofs = fem.locate_dofs_geometrical(V, on_boundary) # 
bc = fem.dirichletbc(default_scalar_type(0), boundary_dofs, V) # Enforces zero displacement on the boundary (denoting that membrane is clamped at edges)

# Weak Formulation and Solution
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = p * v * ufl.dx
problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()

# Interpolating Pressure for Visualization
Q = fem.functionspace(domain, ("Lagrange", 5))
expr = fem.Expression(p, Q.element.interpolation_points())
pressure = fem.Function(Q)
pressure.interpolate(expr)

# Visualiation with PyVista
from dolfinx.plot import vtk_mesh
import pyvista
pyvista.start_xvfb()

# Extract topology from mesh and create pyvista mesh
topology, cell_types, x = vtk_mesh(V)
grid = pyvista.UnstructuredGrid(topology, cell_types, x)

# Set deflection values and add it to plotter
grid.point_data["u"] = uh.x.array
warped = grid.warp_by_scalar("u", factor=25)

plotter = pyvista.Plotter()
plotter.add_mesh(warped, show_edges=True, show_scalar_bar=True, scalars="u")
if not pyvista.OFF_SCREEN:
    plotter.show()
else:
    plotter.screenshot("deflection.png")

# Load Visualization
load_plotter = pyvista.Plotter()
p_grid = pyvista.UnstructuredGrid(*vtk_mesh(Q))
p_grid.point_data["p"] = pressure.x.array.real
warped_p = p_grid.warp_by_scalar("p", factor=0.5)
warped_p.set_active_scalars("p")
load_plotter.add_mesh(warped_p, show_scalar_bar=True)
load_plotter.view_xy()
if not pyvista.OFF_SCREEN:
    load_plotter.show()
else:
    load_plotter.screenshot("load.png")

# Extracting and Plotting Line Data
tol = 0.001  # Avoid hitting the outside of the domain
y = np.linspace(-1 + tol, 1 - tol, 101)
points = np.zeros((3, 101))
points[1] = y
u_values = []
p_values = []

from dolfinx import geometry
bb_tree = geometry.bb_tree(domain, domain.topology.dim)

cells = []
points_on_proc = []
# Find cells whose bounding-box collide with the the points
cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
# Choose one of the cells that contains the point
colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
for i, point in enumerate(points.T):
    if len(colliding_cells.links(i)) > 0:
        points_on_proc.append(point)
        cells.append(colliding_cells.links(i)[0])

points_on_proc = np.array(points_on_proc, dtype=np.float64)
u_values = uh.eval(points_on_proc, cells)
p_values = pressure.eval(points_on_proc, cells)

# Final Plot using Matplotlib
import matplotlib.pyplot as plt
fig = plt.figure()
plt.plot(points_on_proc[:, 1], 50 * u_values, "k", linewidth=2, label="Deflection ($\\times 50$)")
plt.plot(points_on_proc[:, 1], p_values, "b--", linewidth=2, label="Load")
plt.grid(True)
plt.xlabel("y")
plt.legend()
# If run in parallel as a python file, we save a plot per processor
plt.savefig(f"membrane_rank{MPI.COMM_WORLD.rank:d}.png")
