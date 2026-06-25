import os

path = r"C:\Users\SARAN R\.cache\kagglehub\datasets\omkargurav\face-mask-dataset\versions\1\data"

for root, dirs, files in os.walk(path):
    print(root)
    print("Folders:", dirs[:5])
    print("Files:", files[:5])
    print("-" * 50)