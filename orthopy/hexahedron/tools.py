def write(filename, f):
    import meshio
    import meshzoo

    points, cells = meshzoo.cube(-1, +1, -1, +1, -1, +1, 50, 50, 50)
    vals = f(points)
    meshio.write(filename, points, {"tetra": cells}, point_data={"f": vals})
    return
