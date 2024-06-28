import sys
import time

import numpy as np

from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.backends.backend_qtagg import \
    NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.qt_compat import QtWidgets
from matplotlib.figure import Figure


class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self, figures: list[Figure]):
        super().__init__()
        self._main = QtWidgets.QWidget()
        self.setCentralWidget(self._main)
        self.setWindowTitle("Wave function simulation viewer")

        column_height = 2
        column_count = len(figures) // column_height + 1
        layout = QtWidgets.QGridLayout(self._main)
        for c in range(column_count):
            for r in range(column_height):
                if ((column_height * c + r) < len(figures)):
                    fig = figures[column_height * c + r]
                    canvas = FigureCanvas(fig)
                    layout.addWidget(NavigationToolbar(canvas, self), r*2, c)
                    layout.addWidget(canvas, r*2 + 1, c)


def start_app(figures: list[Figure]):
    # Check whether there is already a running QApplication (e.g., if running
    # from an IDE).
    qapp = QtWidgets.QApplication.instance()
    if not qapp:
        qapp = QtWidgets.QApplication(sys.argv)

    app = ApplicationWindow(figures)
    app.showMaximized()
    app.activateWindow()
    app.raise_()
    qapp.exec()