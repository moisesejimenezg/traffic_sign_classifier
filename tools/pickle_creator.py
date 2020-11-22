import pickle
from os import listdir
from PIL import Image
import numpy as np

internet = {"labels": [], "images": [], "file_names": [], "names": []}

valid_images = open("valid_images.txt")
valid_images = valid_images.read()
valid_images = valid_images.split("\n")
valid_images.sort()

folder = "extra_traffic_signs/"
ground_truth = open("extra_traffic_signs/GT-final_test.csv")
ground_truth = ground_truth.read()
ground_truth = ground_truth.split("\n")
ground_truth = [
    int(x.split(";")[7])
    for x in ground_truth
    if x != "" and folder + x.split(";")[0] in valid_images
]

files = listdir("extra_traffic_signs")
files.sort()
for file_name in files:
    if ".ppm" in file_name:
        full_name = "extra_traffic_signs/" + file_name
        image = np.array(Image.open(full_name))
        if image.shape[0] == 32 and image.shape[1] == 32:
            internet["images"].append(image)
            internet["file_names"].append(full_name)
internet["labels"] = ground_truth

signnames = open("signnames.csv")
signnames = signnames.read()
signnames = signnames.split("\n")
signnames = signnames[1:-1]
signnames = [x.split(",")[1] for x in signnames]
internet["names"] = signnames

pickle.dump(internet, open("internet.p", "wb"))
