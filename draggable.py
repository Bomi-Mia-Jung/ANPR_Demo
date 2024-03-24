""" code base courtesy of: https://github.com/yuma-m/matplotlib-draggable-plot/tree/master """
import matplotlib
# matplotlib.use('TkAgg')  # useful backend if you're using Windows
from matplotlib.widgets import Slider

import math
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseEvent

# import personal modules
from lwlr import LWLR, GaussianKernel
from attack import AttractiveAttackOnX


class DraggablePlotTr(object):
    """ A plot with draggable training set """

    class TrPoint:
        def __init__(self, x, y=None):
            self.init_x = float(x)
            self.x = float(x)
            if y:
                self.init_y = float(y)
                self.y = float(y)

    class TePoint:
        def __init__(self, x, y, line=None):
            self.init_x = x
            self.init_y = y
            self.x = x
            self.y = y
            self.init_local_line = line
            self.local_line = line

    def __init__(self, points=None, test_points=None, domain=(0, 100), range=(0,100), r=5, title="Draggable Plot", model=None):
        self._figure, self._axes, self._init_scatterplot, self._scatterplot, self._test_plot, self._init_curve, self._curve, self._radius_circle = None, None, None, None, None, None, None, None

        self._domain = domain
        self._range = range

        self._radius_slider_axes = None
        self._radius_slider = None

        self._r = float(r)

        self._title = title
        if model:
            self._model = model
        else:
            self._init_model()

        self._dragging_point = None
        if points:
            self._points = [self.TrPoint(x=point[0], y=point[1]) for point in points]
        else:
            self._points = []  # list of tuples (x, y)

        self._init_plot()
        self._init_test_points(test_points)
        plt.show()


    def _init_test_points(self, test_points):
        self._test_points = []  # list of TestPoint instances
        if test_points:
            for x_input in test_points:
                self._test_points.append(self.TePoint(x_input, None))

        self._update_plot()

    def _init_plot(self):
        self._figure = plt.figure(self._title)
        axes = plt.subplot(1, 1, 1)
        axes.set_title(self._title)

        # if self._points:
        #     xmin, xmax = float('inf'), -float('inf')
        #     ymin, ymax = float('inf'), -float('inf')
        #
        #     for point in self._points:
        #         xmin = min(xmin, point[0])
        #         xmax = max(xmax, point[0])
        #         ymin = min(ymin, point[0])
        #         ymax = max(ymax, point[0])
        #
        #     axes.set_xlim(xmin, xmax)
        #     axes.set_xlim(ymin, ymax)
        # else:
        axes.set_xlim(self._domain[0], self._domain[1])
        axes.set_ylim(self._range[0], self._range[1])

        # axes.grid(which="both")
        axes.legend(loc='best')
        self._axes = axes

        self._figure.canvas.mpl_connect('button_press_event', self._on_click)
        self._figure.canvas.mpl_connect('button_release_event', self._on_release)
        self._figure.canvas.mpl_connect('motion_notify_event', self._on_motion)

        self._radius_slider_axes = plt.axes([self._domain[0] + 0.15, self._range[0], 0.75, 0.04])
                                        # ([slider_x, slider_y, slider_length, silder_thickness])
        self._radius_slider = Slider(self._radius_slider_axes, 'Radius', self._r, self._r*2)
        self._radius_slider.on_changed(self._update_radius)

    def _update_radius(self, val):
        self._r = val
        if hasattr(self, '_radius_circle') and self._radius_circle:
            self._radius_circle.set_radius(self._r)
            # please leave self.r as argument of set_radius.
            # without this set_radius function, the radius slider feature doesn't work!
            self._figure.canvas.draw_idle()

    def _init_model(self):
        kernel = GaussianKernel
        self._model = LWLR(1, kernel, 2)

    def _update_plot(self):
        if not self._points:
            self._scatterplot.set_data([], [])
            self._curve.set_data([], [])
            for x_test in self._test_points:
                x_test.local_line = None
        else:
            init_X = []
            init_Y = []
            X = [point.x for point in self._points]
            Y = [point.y for point in self._points]

            # Add new plot
            if not self._scatterplot:
                init_X = [point.init_x for point in self._points]
                init_Y = [point.init_y for point in self._points]
                self._init_scatterplot, = self._axes.plot(init_X, init_Y, 'go', markersize=4)
                self._scatterplot, = self._axes.plot(X, Y, 'ro', markersize=4)
            # Update current plot
            else:
                self._scatterplot.set_data(X, Y)

            X = np.array([[x] for x in X])
            Y = np.array([[y] for y in Y])
            x_plot = np.linspace(self._domain[0], self._domain[1], (self._domain[1]-self._domain[0])*2)
            y_plot = np.array(self._model.get_curve(x_plot, X, Y)).squeeze()

            if not self._curve:
                self._curve, = self._axes.plot(x_plot, y_plot, 'r--')
                init_X = np.array([[x] for x in init_X])
                init_Y = np.array([[y] for y in init_Y])
                y_plot = np.array(self._model.get_curve(x_plot, init_X, init_Y)).squeeze()
                self._init_curve, = self._axes.plot(x_plot, y_plot, 'g--')
            else:
                self._curve.set_data(x_plot, y_plot)

            if self._test_points:
                for test_point in self._test_points:
                    test_point.y = self._model(test_point.x, X, Y).item()

                    # plot the local line being learned at the test x_input
                    x_plot = np.linspace(self._domain[0], self._domain[1], self._domain[1] - self._domain[0])
                    y_plot = np.array(self._model.get_local_line(test_point.x, X, Y, x_plot)).squeeze()

                    if not test_point.local_line:
                        line, = self._axes.plot(x_plot, y_plot, '-', label=('locally learned line at x={}'.format(test_point.x)))
                        test_point.local_line = line
                    else:
                        test_point.local_line.set_data(x_plot, y_plot)

                X_test = [test_point.x for test_point in self._test_points]
                Y_test = [test_point.y for test_point in self._test_points]

                if not self._test_plot:
                    self._test_plot, = self._axes.plot(X_test, Y_test, 'x', markersize=7)
                else:
                    self._test_plot.set_data(X_test, Y_test)
            self._axes.legend(loc='best')
            self._figure.canvas.draw()


    def _add_point(self, x, y=None):
        if y is None:
            if isinstance(x, MouseEvent):
                x, y = x.xdata, x.ydata
                point = self.TrPoint(x=x, y=y)
                self._points.append(point)
                return point
        elif y:
            point = self.TrPoint(x=x, y=y)
            self._points.append(self.TrPoint(x=x, y=y))
            return point
        else:
            return

    def _remove_point(self, point):
        # if any(point.x == x and point.y == y for point in self._points):
        #     idx = [(point.x, point.y) for point in self._points].index((x, y))

        # change to remove this instance of TrPoint
        if point in self._points:
            self._points.remove(point)

    def _edit_point(self, point, x, y=None):
        if isinstance(x, MouseEvent):
            x, y = x.xdata, x.ydata
            dist = math.hypot(x - point.init_x, y - point.init_y)

            if dist <= self._r:
                point.x = x
                point.y = y

        else:
            return


    def _find_neighbor_point(self, event):
        """ Find point around mouse position
        :return: TrPoint(x, y) if there are any point around mouse else None
        """
        distance_threshold = 1
        nearest_point = None
        min_distance = float('inf')
        for point in self._points:
            x, y = point.x, point.y
            distance = math.hypot(event.xdata - x, event.ydata - y)
            if distance < min_distance:
                min_distance = distance
                nearest_point = point
        if min_distance < distance_threshold:
            return nearest_point
        return None

    def _draw_circle(self, point):
        # Draw a red circle indicating the allowed movement radius around the point
        if point.init_x is None or point.init_y is None:
            return
        
        if self._radius_circle is not None:
            self._radius_circle.remove()
        radius = self._r
        circle = plt.Circle((point.init_x, point.init_y), radius, color=(1,0.7,0.7), fill=True)
        self._axes.add_patch(circle)
        self._radius_circle = circle

    def _on_click(self, event):
        """ callback method for mouse click event
        :type event: MouseEvent
        """

        # left click
        if event.button == 1 and event.inaxes in [self._axes]:
            point = self._find_neighbor_point(event)
            if point:
                self._dragging_point = point
            else:
                self._add_point(event)
            self._update_plot()

        # right click
        elif event.button == 3 and event.inaxes in [self._axes]:
            point = self._find_neighbor_point(event)
            if point:
                self._remove_point(point)
                self._update_plot()

    def _on_release(self, event):
        """ callback method for mouse release event
        :type event: MouseEvent
        """

        if event.button == 1 and event.inaxes in [self._axes] and self._dragging_point:
            self._dragging_point = None
            """
            Add some way to remove the circle every time it is released :O
            """
            if hasattr(self, '_radius_circle'):
                self._radius_circle.remove()  # Remove the red circle
                self._radius_circle = None  # Set the attribute to None after removal
            self._update_plot()

    def _on_motion(self, event):
        """ callback method for mouse motion event
        :type event: MouseEvent
        """
        if not self._dragging_point:
            return
        if event.xdata is None or event.ydata is None:
            return
        self._edit_point(self._dragging_point, event)
        self._draw_circle(self._dragging_point)
        self._update_plot()


class DraggablePlotTe(DraggablePlotTr):
    """ A plot with draggable target """

    class TargetPoint:
        def __init__(self, x, y):
            self.x = x
            self.y = y
            self._plot_kernel = True  # should the kernel be plotted?
            self._kernel_curve = None  # plot kernel values around target point

    def __init__(self, points=None, test_points=None, domain=(0, 100), range=(0,100), r=5., title="Draggable Plot", model=None, attack=None):
        assert len(test_points) == 1
        if attack:
            self._attack = attack
        else:
            self._set_attack()
        self._target_plot, self._target_point = None, None
        self._curves = []
        super(DraggablePlotTe, self).__init__(points, test_points, domain, range, r, title, model)

    def _update_radius(self, val):
        self._r = val
        self._set_attack()

    def _set_attack(self):
        X = np.array([point.init_x for point in self._points])
        X = np.reshape(X, (X.size, 1))  # (n, d)
        Y = np.array([point.init_y for point in self._points])
        Y = np.reshape(Y, (Y.size, 1))  # (n, 1)
        self._attack = AttractiveAttackOnX(X, Y, self._r, self._model)

    def _update_plot(self):
        if not self._test_points:
            self._test_plot.set_data([], [])
            self._curve.set_data([], [])
        else:
            init_X = np.array([point.init_x for point in self._points])
            init_X = np.reshape(init_X, (init_X.shape[0], 1))
            init_Y = np.array([point.init_y for point in self._points])

            for test_point in self._test_points:
                changed_X, changed_Y = [], []

                if test_point.init_y is None:
                    test_point.init_y = self._model.forward(test_point.x, init_X, init_Y)
                    test_point.y = test_point.init_y
                    self._target_point = self.TargetPoint(x=test_point.init_x, y=test_point.init_y)
                    changed_X, changed_Y = [init_X], [init_Y]
                else:
                    changed_X, changed_Y = [], []
                    for i in range(5):
                        changed_X_, changed_Y_ = self._attack.fit(self._target_point.x, self._target_point.y)
                        changed_X.append(changed_X_)
                        changed_Y.append(changed_Y_)

                    test_point.x = self._target_point.x
                    test_point.init_y = self._model.forward(test_point.x, init_X, init_Y)
                    test_point.y = self._model.forward(test_point.x, changed_X[0], changed_Y[0])

                # print('length of changed_X:', len(changed_X))
                # print('length of changed_X:', len(changed_Y))
                x_plot = np.linspace(self._domain[0], self._domain[1], self._domain[1] - self._domain[0])

                if not self._target_plot:
                    self._target_plot, = self._axes.plot([self._target_point.x], [self._target_point.y], 'r*', markersize=9)
                    if self._target_point._plot_kernel:
                        target_x = np.reshape(np.array([self._target_point.x]), newshape=(1, 1))
                        kernel_x_plot = np.reshape(x_plot, newshape=(x_plot.size, 1))
                        y_plot = self._model.kernel.get_weights(target_x, kernel_x_plot)
                        self._target_point._kernel_curve, = self._axes.plot(x_plot, y_plot, 'm--', alpha=0.7, label='kernel')
                else:
                    self._target_plot.set_data([self._target_point.x], [self._target_point.y])
                    if self._target_point._plot_kernel and self._target_point._kernel_curve:
                        target_x = np.reshape(np.array([self._target_point.x]), newshape=(1, 1))
                        kernel_x_plot = np.reshape(x_plot, newshape=(x_plot.size, 1))
                        y_plot = self._model.kernel.get_weights(target_x, kernel_x_plot)
                        self._target_point._kernel_curve.set_data(x_plot, y_plot)

                # Add new plot
                if not self._scatterplot:
                    self._init_scatterplot, = self._axes.plot(init_X, init_Y, 'go', markersize=4)
                    self._scatterplot, = self._axes.plot(changed_X[0], changed_Y[0], 'ro', markersize=4)
                # Update current plot
                else:
                    self._scatterplot.set_data(changed_X[0], changed_Y[0])

                # plot the local line that was being learned at the test x_input
                y_plot = np.array(self._model.get_local_line(test_point.x, init_X, init_Y, x_plot)).squeeze()

                if not test_point.init_local_line:
                    line, = self._axes.plot(x_plot, y_plot, 'g-', label=('local line before attack at x={:.2f}'.format(self._target_point.x)))
                    test_point.init_local_line = line
                else:
                    test_point.init_local_line.set_data(x_plot, y_plot)
                    test_point.init_local_line.set_label('local line before attack at x={:.2f}'.format(self._target_point.x))

                y_plot = np.array(self._model.get_local_line(test_point.x, changed_X[0], changed_Y[0], x_plot)).squeeze()

                if not test_point.local_line:
                    line, = self._axes.plot(x_plot, y_plot, 'r-', label=('local line after attack at x={:.2f}'.format(self._target_point.x)))
                    test_point.local_line = line
                else:
                    test_point.local_line.set_data(x_plot, y_plot)
                    test_point.local_line.set_label('local line after attack at x={:.2f}'.format(self._target_point.x))

                if not self._test_plot:
                    self._init_pred_y, = self._axes.plot([test_point.x], [test_point.init_y], 'gx', markersize=7)
                    self._test_plot, = self._axes.plot([test_point.x], [test_point.y], 'rx', markersize=7)
                else:
                    self._init_pred_y.set_data([test_point.x], [test_point.init_y])
                    self._test_plot.set_data([test_point.x], [test_point.y])

                x_plot = np.linspace(self._domain[0], self._domain[1], (self._domain[1] - self._domain[0]) * 2)
                y_plot = np.array(self._model.get_curve(x_plot, changed_X[0], changed_Y[0])).squeeze()

                if not self._curve:
                    self._curve, = self._axes.plot(x_plot, y_plot, 'r--')
                    y_plot = np.array(self._model.get_curve(x_plot, init_X, init_Y)).squeeze()
                    self._init_curve, = self._axes.plot(x_plot, y_plot, 'g--')
                else:
                    self._curve.set_data(x_plot, y_plot)

                if len(self._curves) == 0 or len(self._curves) == 1:
                    self._curves = []
                    for i in range(len(changed_X)):
                        y_plot = np.array(self._model.get_curve(x_plot, changed_X[i], changed_Y[i])).squeeze()
                        curve_temp, = self._axes.plot(x_plot, y_plot, 'b--', alpha=0.5)
                        self._curves.append(curve_temp)
                    # print('length of self._curves: ', len(self._curves))
                else:
                    for i in range(len(self._curves)):
                        y_plot = np.array(self._model.get_curve(x_plot, changed_X[i], changed_Y[i])).squeeze()
                        # print(i)
                        self._curves[i].set_data(x_plot, y_plot)

            self._axes.legend(loc='best')
            self._figure.canvas.draw()

    def _edit_point(self, point, x, y=None):
        if isinstance(x, MouseEvent):
            x, y = x.xdata, x.ydata
            point.x, point.y = x, y
        else:
            return

    def _find_neighbor_point(self, event):
        """ Find point around mouse position
        :return: TePoint(x, y) if there are any point around mouse else None
        """
        distance_threshold = 1
        x, y = self._target_point.x, self._target_point.y
        distance = math.hypot(event.xdata - x, event.ydata - y)
        if distance < distance_threshold:
            return self._target_point
        return None

    def _on_click(self, event):
        """ callback method for mouse click event
        :type event: MouseEvent
        """
        # left click
        if event.button == 1 and event.inaxes in [self._axes]:
            point = self._find_neighbor_point(event)
            if point:
                self._dragging_point = point
            self._update_plot()

    def _on_release(self, event):
        """ callback method for mouse release event
        :type event: MouseEvent
        """
        if event.button == 1 and event.inaxes in [self._axes] and self._dragging_point:
            self._dragging_point = None
            self._update_plot()

    def _on_motion(self, event):
        """ callback method for mouse motion event
        :type event: MouseEvent
        """
        if not self._dragging_point:
            return
        if event.xdata is None or event.ydata is None:
            return
        self._edit_point(self._dragging_point, event)
        self._update_plot()

