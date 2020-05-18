def write(filename, f, a=1, n=50):
    import meshio
    import meshzoo

    points, cells = meshzoo.cube(-a, +a, -a, +a, -a, +a, n, n, n)
    vals = f(points)
    meshio.write_points_cells(
        filename, points, {"tetra": cells}, point_data={"f": vals}
    )
