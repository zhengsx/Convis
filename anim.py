import numpy as np
import matplotlib.animation as anim

def update_lines(num, dataLines, lines) :
    lines.set_data(dataLines[0:2,: num])
    lines.set_3d_properties(dataLines[2, :num])
    return lines

def update_points(num, Points, point):
    point.set_data(Points[0, num], Points[1, num])
    point.set_3d_properties(Points[2, num])
    return point