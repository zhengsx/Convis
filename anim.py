import numpy as np
import matplotlib.animation as anim

def update_lines(num, dataLines, lines) :
    lines.set_data(dataLines[0:2,: num])
    lines.set_3d_properties(dataLines[2, :num])
    return lines