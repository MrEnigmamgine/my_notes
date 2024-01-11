import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import colormaps
import math
import numpy as np

def ax_add_circle(radius, deg_step=30, ax=plt.gca()):
    origin = (0,0)
    circle = patches.Circle(origin, radius=radius, facecolor='none', edgecolor='black')
    ax.add_patch(circle)
    for angle in range(0, 360, deg_step):
        x_point = radius*math.cos(math.radians(angle))
        y_point = radius*math.sin(math.radians(angle))
        mag = 1.06
        x_marker = radius*math.cos(math.radians(angle))*mag
        y_marker = radius*math.sin(math.radians(angle))*mag
        ax.plot([0, x_point], [0, y_point], color='black', linestyle='--', alpha=0.2)
        ax.text(x_marker, y_marker, f'{angle}°', ha='center', va='center', color='black', fontsize=8)


def plot_vector_diagram(vectors:list, unitcircle=True, cmap='Dark2'):
    _cmap = colormaps.get(cmap)
    fig, ax = plt.subplots()
    
    
    # Make everything fit on screen and set a 1:1 aspect ratio
    magnitudes = [math.hypot(vector[0],vector[1]) for vector in vectors]
    max = np.max(magnitudes)
    lim=max*1.15
    ax.set_xlim(-lim,lim)
    ax.set_ylim(-lim,lim)
    ax.set_aspect('equal')


    if unitcircle:
        ax_add_circle(max, ax=ax)
    
    #Draw each arrow one at a time with a pretty color
    for i, vector in enumerate(vectors):
        ax.quiver(
            0,
            0,
            vector[0],
            vector[1],
            color= _cmap(i%_cmap.N),
            scale = 1,
            scale_units='xy',
            angles='xy'
        )
    # Pass the new fig, ax objects back in case stuff needs to happen after (Ie. Legends, titles, axis formatting...)
    return fig, ax


plt.close() # Otherwise blank figure shows up after first import