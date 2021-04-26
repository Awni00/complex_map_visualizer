from sympy import *
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors

# experiment w/ styles or create one
plt.style.use('seaborn')


def cmplx_lines_to_tuples(lines):
    """
    converts an array of complex numbers to an array of tuples of (Re, Im)

    Args:
        lines (array): array of sympy complex numbers

    Returns:
        array: array of tuples of floats representing Re and Im components (Re, Im)
    """

    return [[z.evalf().as_real_imag() for z in line] for line in lines]


def apply_func_line(line, func):
    """
    apply a function on a line

    Args:
        line (array): array of complex numbers (input to function)
        func (f:C->C): a function of complex numbers to complex numbers (sympy)

    Returns:
        array: array of points in line w/ func applied to them
    """

    assert np.ndim(line) == 1
    return [func(x).evalf() for x in line]

def plot_lines(lines, color='#269df2', fig_size=(5,5), title=None):
    """plots lines in complelx plane."""

    lines = np.array(lines)
    tuple_lines = cmplx_lines_to_tuples(lines)

    fig, ax = plt.subplots()
    fig.set_size_inches(fig_size)

    for line in tuple_lines:
        x, y = zip(*line)
        ax.plot(x, y, color=color)
    if title is not None: ax.set_title(title)
    ax.set_xlabel('Re')
    ax.set_ylabel('Im')
    ax.set_aspect('equal')

    return fig


def plot_map(lines_in, func, color='#269df2', fig_size=(10,5)):
    """
    given an input space, plots the input space and output space of a complex function.

    Args:
        lines_in (array): array of lines in complex plane
        func (f:C->C): a function from complex nums to complex nums (sympy)
        color (string, optional): color of lines. Defaults to '#269df2'.

    Returns:
        matplotlib.figure.Figure: figure of input space and output space plots
    """

    lines_in = np.array(lines_in)
    tuple_lines_in = cmplx_lines_to_tuples(lines_in)

    lines_out = [apply_func_line(line, func) for line in lines_in]
    tuple_lines_out = cmplx_lines_to_tuples(lines_out)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(fig_size)


    for line_in in tuple_lines_in:
        x, y = zip(*line_in)
        ax1.plot(x, y, color=color)
    ax1.set_title('Input Space')
    ax1.set_xlabel('Re')
    ax1.set_ylabel('Im')
    ax1.set_aspect('equal')

    for line_out in tuple_lines_out:
        x, y = zip(*line_out)
        ax2.plot(x, y, color=color)
    ax2.set_title('Output Space')
    ax2.set_xlabel('Re')
    ax2.set_ylabel('Im')
    ax2.set_aspect('equal')

    return fig


def generate_grid(x_range=(0,1), y_range=(0,1), num_xlines=20,
                    num_ylines=20, line_res=100):
    """
    generates a grid of complex numbers.

    Args:
        x_range (tuple, optional): range of real values. Defaults to (0,1).
        y_range (tuple, optional): range of imaginary values. Defaults to (0,1).
        num_xlines (int, optional): # of horizontal lines in grid. Defaults to 20.
        num_ylines (int, optional): # of vertical lines in grid. Defaults to 20.
        line_res (int, optional): # of points per line. Defaults to 100.

    Returns:
        [array]: array of lines forming a grid
    """

    x_min, x_max = x_range
    y_min, y_max = y_range

    X_hlines = [[x + I*y for x in np.linspace(x_min, x_max, num=line_res)]
                for y in np.linspace(y_min, y_max, num=num_xlines)]
    X_vlines = [[x + I*y for y in np.linspace(x_min, x_max, num=line_res)]
                for x in np.linspace(x_min, x_max, num=num_ylines)]

    X_grid = X_hlines + X_vlines

    return X_grid


def generate_polar_grid(r_max, center=0, num_circles=20, theta_res=200):
    """
    generates a polar grid (concentric circles) of complex numbers

    Args:
        r_max (float): the radius of the final circle.
        center (complex, optional): the center of the polar grid. Defaults to 0.
        num_circles (int, optional): # of circles in grid. Defaults to 20.
        theta_res (int, optional): # of points per circle. Defaults to 200.

    Returns:
        array: array of lines forming a polar grid
    """

    X_circles = [[r*exp(I*theta) + center
                    for theta in np.linspace(0, 2*float(pi), num=theta_res)]
                    for r in np.linspace(0, r_max, num=num_circles)]

    return X_circles

def define_mobius_transform(matrix):
    """
    returns the mobious transform function corresponding to the given matrix

    Args:
        matrix (array): 2x2 array [[a, b], [c, d]]

    Returns:
        function: the corresponding mobious transform function
    """

    (a, b), (c, d) = matrix
    mobius_tfrm = lambda z: (a*z + b) / (c*z + d)
    return mobius_tfrm

# DOMAIN COLOURING

def mag_shading(z_mag):
    '''perform magnitude shading'''
    return 0.5 + 0.5 * (z_mag - np.floor(z_mag))

def grid_lines(z, thresh=0.1):
    '''
    generate gride line shading from complex number outputs.

    Outlines near-integer values of real and imaginary parts.
    '''

    return np.abs(np.sin(np.pi * np.real(z))) ** thresh \
            * np.abs(np.sin(np.pi * np.imag(z))) ** thresh


def domain_color_plot(f, real_lims=(-1,1), imag_lims=(-1,1), base_sat_val=None,
                        mag_as='saturation', gid_lines=True, mag_shading=mag_shading,
                        figsize=(8,8), resolution=(1000,1000), title=None):
    """plots domain colored output space of given function

    Args:
        f (callable): function of complex number. needs to be vectorized (numpy-like)
        real_lims (float tuple, optional): limits of real axis. Defaults to (-1,1).
        imag_lims (float tuple, optional): limits of imaginary axis. Defaults to (-1,1).
        base_sat_val (float, optional): the base value of the remaining colour channel. Defaults to None.
        mag_as (str, optional): the color channel corresponding to complex magnitude. 
        one of 'saturation' or 'value'. Defaults to 'saturation'.
        gid_lines (bool, optional): whether to highlight grid lines. Defaults to True.
        mag_shading (callable, optional): magnitude shading function. Defaults to mag_shading.
        figsize (int tuple, optional): size of figure in inches. Defaults to (8,8).
        resolution (int tuple, optional): resolution of graph per axis. Defaults to (1000,1000).
        title (str, optional): title of graph. Defaults to None.

    Returns:
        (fig, ax) tuple of matplotlib figure and axis
    """

    if base_sat_val == None:
        if mag_as == 'saturation':
            base_sat_val = 1
        elif mag_as == 'value':
            base_sat_val = 0.75

    if mag_as not in ['saturation', 'value']:
        raise ValueError("`mag_as` needs to be one of 'saturation' or 'value'")

    # create input space
    real_space = np.linspace(*real_lims, num=resolution[0])
    imag_space = np.linspace(*imag_lims, num=resolution[1])
    xv, yv = np.meshgrid(real_space, imag_space)
    z_in = xv + yv*1j

    # map to output space
    z_out = f(z_in)

    # color output space
    z_phase = np.angle(z_out, deg=True) # phase
    z_phase = z_phase * (z_phase >= 0) + (360 + z_phase) * (z_phase < 0) # map from (-180, 180] to [0, 360)
    z_phase = z_phase/360 # normalized phase as hue

    z_mag = np.abs(z_out) # magnitude


    if mag_shading:
        z_mag = mag_shading(z_mag) # perform magnirude shading
    else:
        z_mag = z_mag / np.max(z_mag) # normalized magnitude as value

    if grid_lines:
        value = grid_lines(z_out)
    else:
        value = np.ones_like(z_phase)*base_sat_val

    if mag_as == 'saturation':
        hsv_color = np.stack([z_phase, z_mag, value], axis=-1) # create hsv image
    elif mag_as == 'value':
        hsv_color = np.stack([z_phase, value, z_mag], axis=-1) # create hsv image


    rgb_color = matplotlib.colors.hsv_to_rgb(hsv_color)

    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    ax.imshow(rgb_color, aspect='equal', extent=(*real_lims, *imag_lims))
    ax.set_xlabel('Re')
    ax.set_ylabel('Im')
    if title: ax.set_title(title)

    return fig, ax