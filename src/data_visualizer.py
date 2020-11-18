import matplotlib.pyplot as plt
import numpy as np


def fill_empty_position(axes, x, y, img):
    axes[x, y].imshow(255 + np.zeros(img.squeeze().shape))
    axes[x, y].set_axis_off()


class DataVisualizer:
    def __init__(self, data, test_labels, valid_labels, train_labels, class_number):
        self.class_number = class_number
        self.data = data
        self.test_labels = test_labels
        self.valid_labels = valid_labels
        self.train_labels = train_labels
        self.labels = np.concatenate((test_labels, valid_labels, train_labels))

    def generate_histogram(self):
        plt.hist(
            [self.test_labels, self.valid_labels, self.train_labels],
            label=["Test", "Validation", "Train"],
            bins=range(0, 43),
        )
        plt.legend(loc="upper center")
        plt.title("Label Histogram")

    def visualize(self):
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
        f.set_figheight(8)
        f.set_figwidth(8)
        fill_empty_position(axes, 8, 3, self.data[0])
        fill_empty_position(axes, 8, 4, self.data[0])
        plt.show()
