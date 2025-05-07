# Part B: Mesh Refinement Study (Beam Deflection)

Based on this example:
![image](https://github.com/user-attachments/assets/3a0d3936-20fa-455d-83c2-e22ef42324b3)


## Library Imports
```python
# Necessary Library Imports
import pyvista                              # 3D visualization (not used here but could be useful for future extensions)
from dolfinx import mesh, fem, plot, io, default_scalar_type  # DOLFINx core modules
from dolfinx.fem.petsc import LinearProblem # High-level FEM problem solving
from mpi4py import MPI                      # For parallel computation
import ufl                                  # Unified Form Language (for variational forms)
import numpy as np                          # Numerical computing
import matplotlib.pyplot as plt             # Plotting library
import time                                 # Optional timing tool
```

## Problem Setup
```python
# Problem Parameters/Constants
L = 1             # Length of the beam
W = 0.2           # Width and height of the beam
mu = 1            # Shear modulus
rho = 1           # Density
delta = W / L     # Aspect ratio
gamma = 0.4 * delta**2  # Scaling factor for gravity
beta = 1.25       # Poisson-related constant (used as lambda)
lambda_ = beta    # First Lamé parameter
g = gamma         # Gravitational acceleration (scaled)


# Results Storage
displace_list_1 = []  # Stores max displacement for first-order elements
displace_list_2 = []  # Stores max displacement for second-order elements
mesh_list = []        # Stores mesh refinements (number of elements along length)
```

## Main Loop for Mesh Generation and Solving Process
```python
# Loop over element orders: 1 = linear, 2 = quadratic
for order in range(1, 3):

    # Timing the mesh generation and solving process
    # print(f"Processing mesh with {j} nodes and order {order}...")
    # start_time = time.time()
    
    # Loop over mesh refinements along beam length
    for j in range(2, 25):
        # Create structured hexahedral mesh of the beam domain
        domain = mesh.create_box(MPI.COMM_WORLD,
                                 [np.array([0, 0, 0]), np.array([L, W, W])],
                                 [j, 6, 6],  # refine only in the x-direction
                                 cell_type=mesh.CellType.hexahedron)

        # Define vector-valued Lagrange function space (3D displacements)
        V = fem.functionspace(domain, ("Lagrange", order, (domain.geometry.dim,)))

        # Define clamped boundary condition at x = 0 (left face of beam)
        def beam_bc(x):
            return np.isclose(x[0], 0)

        fdim = domain.topology.dim - 1  # Facet dimension
        boundary_facets = mesh.locate_entities_boundary(domain, fdim, beam_bc)
        u_D = np.array([0, 0, 0], dtype=default_scalar_type)  # Zero displacement vector
        bc = fem.dirichletbc(u_D,
                             fem.locate_dofs_topological(V, fdim, boundary_facets),
                             V)

        # Define external surface traction
        T = fem.Constant(domain, default_scalar_type((0, 0, 0)))
        ds = ufl.Measure("ds", domain=domain)

        # Define strain and stress tensors using linear elasticity
        def epsilon(u):
            return ufl.sym(ufl.grad(u))  # Symmetric gradient

        def sigma(u):
            return lambda_ * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2 * mu * epsilon(u)

        # Variational problem definition (weak formulation)
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        f = fem.Constant(domain, default_scalar_type((0, 0, -rho * g)))  # Body force (gravity)
        a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx                    # Bilinear form
        L_form = ufl.dot(f, v) * ufl.dx + ufl.dot(T, v) * ds            # Linear form

        # Solve linear system using LU factorization
        problem = LinearProblem(a, L_form, bcs=[bc],
                                petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        uh = problem.solve()  # uh is the displacement field (Function)

        # Time taken to solve the problem
        # print(f"Time to solve mesh {j} order {order}: {time.time() - start_time:.2f} seconds")

        # Post-Processing: Max Displacement
        u_array = uh.x.array.reshape((-1, 3))          # Displacement vectors at DOFs
        u_magnitude = np.linalg.norm(u_array, axis=1)  # Compute magnitude of displacement
        max_deflection = np.max(u_magnitude)           # Find max value
        max_index = np.argmax(u_magnitude)             # Index of max displacement

        # Coordinates of max displacement point
        dof_coordinates = V.tabulate_dof_coordinates()
        max_point_coords = dof_coordinates[max_index]

        # Store results for later plotting
        if order == 1:
            displace_list_1.append(max_deflection)
            mesh_list.append(j)  # Save current mesh refinement level
        else:
            displace_list_2.append(max_deflection)
```

## Plotting Mesh Convergence
```python
# Plotting Mesh Refinement Study

# Compare max displacement vs. mesh refinement for the two polynomial orders
plt.plot(mesh_list, displace_list_1, label='First Order Elements', color='red')
plt.plot(mesh_list, displace_list_2, label='Second Order Elements', color='blue')
plt.title("Mesh Refinement")                 # Plot title
plt.xlabel("Number of Elements (Length)")         # x-axis label
plt.ylabel("Max Displacement")               # y-axis label
plt.legend(loc='lower right')                # Legend placement
plt.savefig("Mesh Refinement.png", dpi=300)  # Save plot to PNG
plt.close()                                  # Close the plot
```
![image](https://github.com/user-attachments/assets/dee4eedf-de85-409e-a905-9141c3c6709d)


