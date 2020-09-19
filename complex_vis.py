from sympy import *
import numpy as np

import matplotlib.pyplot as plt

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
    fig.set_size_inches([10, 5])


    for line_in in tuple_lines_in:
        ax1.set_title('Input Space')
        ax1.set_xlabel('Re')
        ax1.set_ylabel('Im')

        x, y = zip(*line_in)
        ax1.plot(x, y, color=color)

    for line_out in tuple_lines_out:
        ax2.set_title('Output Space')
        ax2.set_xlabel('Re')
        ax2.set_ylabel('Im')

        x, y = zip(*line_out)
        ax2.plot(x, y, color=color)

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
