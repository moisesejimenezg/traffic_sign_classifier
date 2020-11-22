import matplotlib.pyplot as plt
from sys import argv

image_path = argv[1]

plt.figure(figsize=(2,2))
plt.imshow(plt.imread(image_path))
plt.axis("off")
plt.show()
