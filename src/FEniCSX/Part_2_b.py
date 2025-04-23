# Mesh Refinement Study

def run_simulation(mesh_size=0.05, poly_order=1, label="original"):
    import gmsh
    from mpi4py import MPI
    import numpy as np
    import ufl
    from dolfinx import fem, geometry, default_scalar_type
    from dolfinx.io import gmshio
    from dolfinx.fem.petsc import LinearProblem
    import pyvista
    import matplotlib.pyplot as plt
    from dolfinx.plot import vtk_mesh

    gmsh.initialize()
    gmsh.model.occ.addDisk(0, 0, 0, 1, 1)
    gmsh.model.occ.synchronize()
    gdim = 2
    gmsh.model.addPhysicalGroup(gdim, [1], 1)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
    gmsh.model.mesh.generate(gdim)

    domain, cell_markers, facet_markers = gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=gdim)
    mpirun -n 4 python your_script.py
    gmsh.finalize()

    V = fem.functionspace(domain, ("Lagrange", poly_order))
    x = ufl.SpatialCoordinate(domain)
    beta = fem.Constant(domain, default_scalar_type(12))
    R0 = fem.Constant(domain, default_scalar_type(0.3))
    p = 4 * ufl.exp(-beta**2 * (x[0]**2 + (x[1] - R0)**2))

    def on_boundary(x):
        return np.isclose(np.sqrt(x[0]**2 + x[1]**2), 1)

    boundary_dofs = fem.locate_dofs_geometrical(V, on_boundary)
    bc = fem.dirichletbc(default_scalar_type(0), boundary_dofs, V)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = p * v * ufl.dx
    problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uh = problem.solve()

    Q = fem.functionspace(domain, ("Lagrange", 5))
    expr = fem.Expression(p, Q.element.interpolation_points())
    pressure = fem.Function(Q)
    pressure.interpolate(expr)

    # Extract line data
    tol = 0.001
    y = np.linspace(-1 + tol, 1 - tol, 101)
    points = np.zeros((3, 101))
    points[1] = y
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)

    points_on_proc = []
    cells = []
    for i, point in enumerate(points.T):
        if len(colliding_cells.links(i)) > 0:
            points_on_proc.append(point)
            cells.append(colliding_cells.links(i)[0])

    points_on_proc = np.array(points_on_proc, dtype=np.float64)
    u_values = uh.eval(points_on_proc, cells)
    p_values = pressure.eval(points_on_proc, cells)

    # Plot
    fig = plt.figure()
    plt.plot(points_on_proc[:, 1], 50 * u_values, "k", linewidth=2, label=f"Deflection ×50 ({label})")
    plt.plot(points_on_proc[:, 1], p_values, "b--", linewidth=2, label="Load")
    plt.grid(True)
    plt.xlabel("y")
    plt.legend()
    plt.savefig(f"membrane_{label}_rank{MPI.COMM_WORLD.rank:d}.png")
    plt.close()

run_simulation(mesh_size=0.05, poly_order=1, label="original")
run_simulation(mesh_size=0.025, poly_order=1, label="h-refined")
run_simulation(mesh_size=0.05, poly_order=2, label="p-refined")
