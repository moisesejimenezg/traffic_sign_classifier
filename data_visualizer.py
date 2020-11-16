import matplotlib.pyplot as plt
import numpy as np

def fill_empty_position(axes, x, y, img):
    axes[x, y].imshow(np.zeros(img.squeeze().shape))
    axes[x, y].get_xaxis().set_visible(False)
    axes[x, y].get_yaxis().set_visible(False)


class DataVisualizer:
    def __init__(self, data, labels, class_number):
        self.class_number = class_number
        self.data = data
        self.labels = labels

    def visualize(self):
        self.__generate_histogram()
        positions = self.__find_examples()
        self.__display_grid(positions)

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

    def __display_grid(self, positions):
        rows = 9
        cols = 5
        i = 0
        f, axes = plt.subplots(rows, cols)
        for position in positions:
            x = i // cols
            y = i % cols
            axes[x, y].imshow(self.data[position].squeeze())
            axes[x, y].get_xaxis().set_visible(False)
            axes[x, y].get_yaxis().set_visible(False)
            i += 1
        fill_empty_position(axes, 8, 3, self.data[0])
        fill_empty_position(axes, 8, 4, self.data[0])
        plt.show()

    def __generate_histogram(self):
        plt.hist(self.labels, range(0, 43))
