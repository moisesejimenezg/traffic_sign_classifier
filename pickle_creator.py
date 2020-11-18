import pickle
from os import listdir
from PIL import Image
import numpy as np
internet = {"labels":[], "images": [], "names": []}

for file_name in listdir("extra_traffic_signs"):
    if ".ppm" in file_name:
        full_name = "extra_traffic_signs/" + file_name
        image = np.array(Image.open(full_name))
        if image.shape[0] == 32 and image.shape[1] == 32:
            internet["images"].append(image)
            internet["names"].append(full_name)
            print(full_name)

pickle.dump(internet, open("internet.p", "wb"))
