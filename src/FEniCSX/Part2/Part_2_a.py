# Part A: Problem with Known Analytical Solution (Poisson Equation)

# Imports for specific functions
from mpi4py import MPI # Message Passing Interface, required for parallel processing
from dolfinx import mesh # Mesh generation class within DolfinX

# Define discrete domain (construct singular mesh, parallel processing)
domain = mesh.create_unit_square(MPI.COMM_WORLD, 8, 8, mesh.CellType.quadrilateral) # Creates a square mesh from 8 x 8 quadrilateral elements, mesh generation is distributed across all available processors
mpirun -n 2 Part_2_a.py # Alters above command, script will specifically use 2 processors

# Create finite element function space
from dolfinx.fem import functionspace
V = functionspace(domain, ("Lagrange", 1)) # Defines a function space over the domain, using Langrange elements of degree 1 (piecewise linear elements)
from dolfinx import fem
# Function definition
uD = fem.Function(V) # Specifies Finite Element Function over space V
uD = interpolate(lambda x: 1 + x[0]**2 + 2 * x[1]**2) # Interpolate is used to set the the FE function to the known analytical solution

import numpy
# Create facet to cell connectivity required to determine boundary facets
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim) # Creates connectivity between facets (faces of the mesh) and individual cells
boundary_facets = mesh.exterior_facet_indices(domain.topology) # Identifies the exterior boundary faces where boundary conditions will be applied

# Create Dirichlet Boundary Condition
boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets) # Finds the degrees of freedom on the boundary faces
bc = fem.dirichletbc(uD, boundary_dofs) # Applies the boundary conditions (where unknown function = uD) at the boundary

# Define Trial and Test functions
import ufl
u = ufl.TrialFunction(V) # Trial Function (unknown solution that is being solved for)
v = ufl.TestFunction(V) # Test Function (function used to multiply the actual function, used for weak formulation of the equation)

# Define Source Term
from dolfinx import default_scalar_type 
f = fem.Constant(domain, default_scalar_type(-6)) # Defines constant value for source term ((fx) = -6), used in the weak formulation

# Define Variational Problem (Weak Formulation, multiply source function by test function)
a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx # Left side of equation
L = f * v * ufl.dx # Right side of equation


from dolfinx.fem.petsc import LinearProblem
problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"}) # LinearProblem solves the linear system for the weak formuation (PETSc)
uh = problem.solve() # Numerical solution of the Poisson equation after solving the linear problem

# Computing Solution Error (Validation Step)
V2 = fem.functionspace(domain, ("Lagrange", 2)) # Defines higher-degree function space (L^2 elements, degree 2 instead of degree 1) for the exact solution
uex = fem.Function(V2) # Defines Finite Element Function over space V2
uex.interpolate(lambda x: 1 + x[0]**2 + 2 * x[1]**2) # Interpolates the analytical solution into the higher-order function space

L2_error = fem.form(ufl.inner(uh - uex, uh - uex) * ufl.dx) # Defines error in L^2 norm by the difference of the numerical solution (uh) and the exact solution (uex)
error_local = fem.assemble_scalar(L2_error)
error_L2 = numpy.sqrt(domain.comm.allreduce(error_local, op=MPI.SUM)) # Reduce error accross all processors (in parallel) by summing up the error from each processed run
error_max = numpy.max(numpy.abs(uD.x.array-uh.x.array)) # Computes maximum pointwise error between boundary condition and solution

# Print the L^2 error and maximum error on one processor
if domain.comm.rank == 0:
    print(f"Error_L2 : {error_L2:.2e}")
    print(f"Error_max : {error_max:.2e}")

# Plot mesh using pyvista (Python library for 3D visualization)
import pyvista 
print(pyvista.global_theme.jupyter_backend)

from dolfinx import plot
pyvista.start_xvfb()
domain.topology.create_connectivity(tdim, tdim)
topology, cell_types, geometry = plot.vtk_mesh(domain, tdim) # Converts mesh into plottable format for PyVista
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

# Visualise Mesh
plotter = pyvista.Plotter()
plotter.add_mesh(grid, show_edges=True) # Adds mesh to plot and visualizes it
plotter.view_xy()
if not pyvista.OFF_SCREEN:
    plotter.show()
else:
    figure = plotter.screenshot("fundamentals_mesh.png")


# Plot function using pyvista
u_topology, u_cell_types, u_geometry = plot.vtk_mesh(V) 
u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
u_grid.point_data["u"] = uh.x.array.real # Assigns computed solution values to the mesh
u_grid.set_active_scalars("u")
u_plotter = pyvista.Plotter()
u_plotter.add_mesh(u_grid, show_edges=True) # Plots the solution using PyVista
u_plotter.view_xy()
if not pyvista.OFF_SCREEN:
    u_plotter.show()

# Warp mesh by scalar factor for 3D visualization
warped = u_grid.warp_by_scalar() # Creates a 3D effect by warping the mesh by the solution values
plotter2 = pyvista.Plotter()
plotter2.add_mesh(warped, show_edges=True, show_scalar_bar=True)
if not pyvista.OFF_SCREEN:
    plotter2.show()
