import matplotlib.pyplot as plt
import numpy as np


class DataVisualizer:
    def __init__(self, data, labels, class_number):
        self.class_number = class_number
        self.data = data
        self.labels = labels

    def visualize(self):
        self.__generate_histogram()
        positions = self.__find_examples()
        for position in positions:
            self.__display_image(position)

    def __find_examples(self):
        added = 0
        positions = [-1] * self.class_number
        i = 0
        while added < self.class_number:
            idx = self.labels[i]
            if positions[idx] == -1:
                positions[idx] = i
                added += 1
            i += 1
        return positions

    def __display_image(self, idx):
        plt.imshow(self.data[idx].squeeze())
        plt.show()

    def __generate_histogram(self):
        # histogram = np.histogram(self.labels, range(0, 43))
        plt.hist(self.labels, range(0, 43))
        plt.show()
