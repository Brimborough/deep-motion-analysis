import csv
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D

class AnimateElderlyData(object):
    def __init__(self, file_name):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')

        scale = 5
        self.ax.set_xlim3d(-scale*30, scale*30)
        self.ax.set_zlim3d( 0, scale*60)
        self.ax.set_ylim3d(-scale*30, scale*30)

        self.line = self.read_csv_data(file_name)

        for i in xrange(5):
            self.line.next()

        #check whether the current column is a position or rotation data
        self.position_array = self.line.next()

        #check whether the current column is x, y, or z
        self.xyz_array = self.line.next()

        x, y, z = self.data_stream()

        self.points = plt.plot(x, z, y, 'o')[0]
        self.ani = animation.FuncAnimation(self.fig, self.update, np.arange(1000), interval=30)

        self.ax.set_xlabel('X Label')
        self.ax.set_ylabel('Y Label')
        self.ax.set_zlabel('Z Label')

    def read_csv_data(self, file_name):
        with open(file_name, 'rU') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')

            for row in reader:
                yield row

    def data_stream(self):
        x_list = []
        y_list = []
        z_list = []

        idx = 0
        row = self.line.next()

        for col_val in row:
            if self.position_array[idx] == "Position" and col_val:
                if self.xyz_array[idx] == "X":
                    x_list.append(float(col_val))
                elif self.xyz_array[idx] == "Y":
                    y_list.append(float(col_val))
                elif self.xyz_array[idx] == "Z":
                    z_list.append(float(col_val))
                else:
                    raise ValueError('Invalid file formating')

                idx += 1

        return x_list, y_list, z_list

    def update(self, i):
        x, y, z = self.data_stream()

        self.points.set_data(x, z)
        self.points.set_3d_properties(y)

        return self.points

    def show(self):
        plt.show()
