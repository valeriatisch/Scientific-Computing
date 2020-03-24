
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

'''
    This package is to be used as a library. Please do not edit.
'''


def runge_function(n: int = 100, min_x: float = -5.0, max_x: float = 5.0) -> (np.ndarray, np.ndarray):
    """
    Compute the discrete Runge function on the linearly spaced inverval [min_x, max_x] with n function values.

    Arguments:
    min_x: left border of the interval
    max_x: right border of the interval
    n: number of function values inside the interval

    Return:
    x: vector containing all x values, correspond to values in y
    y: vector containing all function values, correspond to values in x
    """

    x = np.linspace(min_x, max_x, n)
    y = 1.0 / (1.0 + x ** 2)

    return x, y


def pad_coefficients(poly, length):
    """Adds zeros to the coefficients of poly if they have not the proper length."""
    return np.pad(poly.coeffs, (length - poly.coeffs.size, 0), mode='constant', constant_values=0)


def plot_function(x, y):
    """ Plot the function that is given by the discrete point pairs in x and y. """
    plt.grid(True)
    plt.plot(x, y, 'r-')
    min_x = np.min(x)
    min_y = np.min(y)
    max_x = np.max(x)
    max_y = np.max(y)
    scale_x = max_x - min_x
    scale_y = max_y - min_y
    plt.xlim(min_x - 0.05 * scale_x, max_x + 0.05 * scale_x)
    plt.ylim(min_y - 0.05 * scale_y, max_y + 0.05 * scale_y)
    plt.show()


def plot_function_interpolations(function, support_points, interpolations, bases):
    """ Plot a grid with the given function, the support points, interpolation and bases in each plot. """
    x_f, y_f = function
    fig1 = plt.figure()
    for i in range(len(support_points)):
        x_s, y_s = support_points[i]
        x_i, y_i = interpolations[i]
        p = fig1.add_subplot(3, 3, i + 1)
        p.grid(True)
        p.set_xlim(-5.3, 5.3)
        p.set_xticks([-5, 0, 5])
        p.set_ylim(-1.2, 2.2)
        p.plot(x_f, y_f, 'r-')
        p.plot(x_s, y_s, 'ko')
        p.plot(x_i, y_i, 'b-')

    fig2 = plt.figure()
    for i in range(len(bases)):
        p1 = fig2.add_subplot(3, 3, i + 1)
        p1.grid(True)
        p1.set_xlim(-5.3, 5.3)
        p1.set_xticks([-5, 0, 5])
        p1.set_ylim(-1.2, 2.2)
        for base_func in bases[i]: plt.plot(x_f, base_func(x_f), '-')

    plt.show()


def plot_spline(points, interpolations):
    """ Plot a spline with the interpolation points."""

    # Plot interpolation points
    plt.plot(points[0], points[1], 'ko')

    # Plot piecewise interpolants
    for i in range(len(points[0]) - 1):
        # plot local interpolant
        p = interpolations[i]
        px = np.linspace(points[0][i], points[0][i + 1], 100 / len(points[0]))
        py = p(px)
        plt.plot(px, py, '-')

    # Plot Runge function
    rx = np.linspace(-5, 5, 100)
    ry = 1.0 / (1 + rx ** 2)
    plt.plot(rx, ry, '--', color='0.7')

    # Beautify plot
    plt.grid(True)
    plt.xlim(-5.1, 5.1)
    plt.xticks(np.linspace(-5, 5, 11))
    plt.ylim(-0.1, 1.1)
    plt.subplots_adjust(left=0.05, right=0.98, top=0.98, bottom=0.05)

    plt.show()


class Stickguy:
    """ The stick guy. Only use in this package. """

    def __init__(self, ax):
        self.spine, = ax.plot([], [], lw=2)
        self.left_arm, = ax.plot([], [], lw=2)
        self.right_arm, = ax.plot([], [], lw=2)
        self.left_leg, = ax.plot([], [], lw=2)
        self.right_leg, = ax.plot([], [], lw=2)


def linear_animation(keytime, keyframe):
    """
    The returned function computes interpolated keyframe curframe at given time t.
    It uses the given keytime and splines parameters for this.
    """

    def animation_function(t):
        k = np.searchsorted(keytime, t, side='right') - 1
        u = (t - keytime[k]) / (keytime[k + 1] - keytime[k])
        curframe = (1.0 - u) * keyframe[k] + u * keyframe[k + 1]
        return curframe

    return animation_function


def cubic_animation(keytime, splines):
    """
    The returned function computes interpolated keyframe curframe at given time t.
    It uses the given keytime and splines parameters for this.
    """

    def animation_function(t):
        k = np.searchsorted(keytime, t, side='right') - 1
        curframe = np.array([s[k](t) for s in splines])

        return curframe

    return animation_function


def param2pos(param, stickguy):
    """
    Computes positions of joints for the stick guy.
    Inputs:
    param : list of parameters describing the pose
    param[0]: height of hip
    param[1]: angle of spine to vertical axis
    param[2]: angle of upper arm 0 to spine
    param[3]: angle of lower arm 0 to upper arm 0
    param[4,5]: as above, other arm
    param[6]: angle of neck/head to spine
    param[7]: angle of upper leg 0 to vertical axis
    param[8]: angle of lower leg 0 to upper leg 0
    param[9,10]: as above, other leg
    """

    hip_pos = np.array([0.0, param[0]])
    spine_vec = np.array([0.0, 1.0])
    spine_vec = rotate(spine_vec, param[1])
    neck_pos = hip_pos + spine_vec
    basic_arm_vec = -0.6 * spine_vec
    arm_vec = rotate(basic_arm_vec, param[2])
    left_elbow_pos = neck_pos + arm_vec
    arm_vec = rotate(arm_vec, param[3])
    left_hand_pos = left_elbow_pos + arm_vec
    lad = np.array([neck_pos, left_elbow_pos, left_hand_pos])
    stickguy.left_arm.set_data(lad[:, 0], lad[:, 1])

    arm_vec = rotate(basic_arm_vec, param[4])
    right_elbow_pos = neck_pos + arm_vec
    arm_vec = rotate(arm_vec, param[5])
    right_hand_pos = right_elbow_pos + arm_vec
    rad = np.array([neck_pos, right_elbow_pos, right_hand_pos])
    stickguy.right_arm.set_data(rad[:, 0], rad[:, 1])

    neck_vec = 0.3 * spine_vec
    neck_vec = rotate(neck_vec, param[6])
    head_pos = neck_pos + neck_vec
    sd = np.array([hip_pos, neck_pos, head_pos])
    stickguy.spine.set_data(sd[:, 0], sd[:, 1])

    basic_leg_vec = (0.0, -0.7)
    leg_vec = rotate(basic_leg_vec, param[7])
    left_knee_pos = hip_pos + leg_vec
    leg_vec = rotate(leg_vec, param[8])
    left_foot_pos = left_knee_pos + leg_vec
    lld = np.array([hip_pos, left_knee_pos, left_foot_pos])
    stickguy.left_leg.set_data(lld[:, 0], lld[:, 1])

    leg_vec = rotate(basic_leg_vec, param[9])
    right_knee_pos = hip_pos + leg_vec
    leg_vec = rotate(leg_vec, param[10])
    right_foot_pos = right_knee_pos + leg_vec
    rld = np.array([hip_pos, right_knee_pos, right_foot_pos])
    stickguy.right_leg.set_data(rld[:, 0], rld[:, 1])

    return


def rotate(v, angle):
    """ Helper function to turn a vector for a given angle. """
    s = np.sin(angle)
    c = np.cos(angle)
    rv = np.array([v[0] * c - v[1] * s, v[0] * s + v[1] * c])
    return rv


def animate(keytime, keyframe, interpolate):
    """ Animates the stickguy with the given interpolation function and frames. """
    # Definitonen der Figur und des Plots.
    fig = plt.figure(figsize=(15, 5))
    ax = plt.subplot2grid((1, 4), (0, 0))
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-2, 2)
    stickguy = Stickguy(ax)

    px = plt.subplot2grid((1, 4), (0, 1), colspan=3)
    px.set_xlim(0, 200)
    px.set_ylim(-0.4, 0.4)
    curves = [px.plot([], [], "-")[0] for i in range(11)]

    def anim(t):
        curframe = interpolate(t)
        param2pos(curframe, stickguy)
        global curves_x, curves_y
        if t == 0:
            curves_x = [float(t)]
            curves_y = curframe
        else:
            curves_x.append(float(t))
            curves_y = np.c_[curves_y, curframe]
        for i in range(len(curves)):
            curves[i].set_data(curves_x, curves_y[i])

        return stickguy.left_arm, stickguy.right_arm, stickguy.spine, stickguy.left_leg, stickguy.right_leg

    # Die Animation vom Stick Guy
    anim = animation.FuncAnimation(fig, anim, frames=200, interval=50, blit=False)
    plt.subplots_adjust(left=0.04, right=0.98, top=0.98, bottom=0.05)
    plt.show()
