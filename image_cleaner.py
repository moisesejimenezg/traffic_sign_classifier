import os

valid_images = open("valid_images.txt")
valid_images = valid_images.read()
valid_images = valid_images.split("\n")
to_delete = []
folder = "extra_traffic_signs/"

for file_name in os.listdir("extra_traffic_signs"):
    full_name = folder + file_name
    if (
        full_name not in valid_images
        and full_name != "extra_traffic_signs/GT-final_test.csv"
    ):
        to_delete.append(full_name)

for full_name in to_delete:
    os.remove(full_name)
