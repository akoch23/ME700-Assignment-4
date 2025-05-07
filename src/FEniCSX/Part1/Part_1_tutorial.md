# Tutorial for Part 1 Code (3D Membrane Deflection Problem)

This tutorial will cover the function and expect output of Part_1_Example.py within this repository, based on the implementation guide of similar name from the official FEniCSx tutorial handbook.

## Creating the Mesh
To create the digital geometry of the desired shape, the Python-API of GMSH is used. 

``` python
# Mesh Generation for Model (2D Circular Disk)
gmsh.initialize()  # Initialize the Gmsh API session
membrane = gmsh.model.occ.addDisk(0, 0, 0, 1, 1) # Create a 2D disk (x, y, z, x-radius, y-radius) with radius 1 centered at origin
gmsh.model.occ.synchronize() # Finalize CAD operations and synchronize the model
gdim = 2 # Geometric dimension for 2D model
gmsh.model.addPhysicalGroup(gdim, [membrane], 1) # Assign physical group for FEM tagging
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.05) # Set mesh resolution (min)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.05) # Set mesh resolution (max)
gmsh.model.mesh.generate(gdim) # Generate 2D mesh
print("GMSH mesh generated.")
```
