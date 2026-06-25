import os
import kagglehub

path = kagglehub.dataset_download(
    "omkargurav/face-mask-dataset"
)

print(path)

for item in os.listdir(path):
    print(item)