import os
import csv
import numpy as np
import cv2
from skimage.feature import canny
from scipy import ndimage as ndi
import math

def image_prep(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    binary_img = img > 128  # Simple threshold to make it binary (adjust 128 if needed)
    fill_img = ndi.binary_fill_holes(binary_img) # Fill holes within cells
    labeled_img, num_features = ndi.label(fill_img) # Label connected components (cells)
    return labeled_img, num_features

def numofneighbour(mat, i, j, searchValue):
    count = 0
    if i > 0 and mat[i - 1][j] == searchValue:
        count += 1
    if j > 0 and mat[i][j - 1] == searchValue:
        count += 1
    if i < len(mat) - 1 and mat[i + 1][j] == searchValue:
        count += 1
    if j < len(mat[i]) - 1 and mat[i][j + 1] == searchValue:
        count += 1
    return count

def findperimeter(mat, num_features):
    perimeter = [0] * (num_features + 1)
    for i in range(len(mat)):
        for j in range(0, len(mat[i])):
            if mat[i][j] != 0:
                perimeter[mat[i][j]] += (4 - numofneighbour(mat, i, j, mat[i][j]))
    return perimeter

def extract_area_perim(img,num_features):
    area = [0] * (num_features+1)
    for i in range(len(img)):
        for j in range(len(img[i])):
            value = img[i][j]
            if(value != 0):
                area[value]+=1
    return area, findperimeter(img,num_features)

def extract_circularity(area, perimeter):
    if not area or not perimeter:
        return []
    circularity = []
    for i in range(len(area)):
        if perimeter[i] != 0:
            circularity.append((4 * math.pi * area[i]) / (math.pow(perimeter[i], 2)))
        else:
            circularity.append(0)
    return circularity

def convert_to_relative(cellAreas, cellPerims):
    if not cellAreas or not cellPerims:
        return cellAreas, cellPerims
    relativeArea = []
    relativePerim = []
    if cellAreas:
        averageCellSize = sum(cellAreas) / len(cellAreas) if len(cellAreas) > 0 else 1
        relativeArea = [area / averageCellSize for area in cellAreas]
    if cellPerims:
        averagePerimSize = sum(cellPerims) / len(cellPerims) if len(cellPerims) > 0 else 1
        relativePerim = [perim / averagePerimSize for perim in cellPerims]
    return relativeArea, relativePerim

def removeFromImg(img, feature):
    for i in range(len(img)):
        for j in range(0, len(img[i])):
            if (img[i][j] == feature):
                img[i][j] = 0
    return img

def removeOutliers(area, perim, img):
    if not area or not perim:
        return [], [], img
    mean = np.mean(area)
    std = np.std(area)
    new_area = []
    new_perim = []
    new_img = img.copy()
    for i in range(len(area)):
        if area[i] <= mean + 3.5 * std:
            new_area.append(area[i])
            new_perim.append(perim[i])
        else:
            print("Popping feature:", i + 1)
            new_img = removeFromImg(new_img, i + 1)
    if len(new_area) == 0 and len(area) > 0: # Check if original area list was not empty before trying to get max
        max_area_index = np.argmax(area)
        new_area.append(area[max_area_index])
        new_perim.append(perim[max_area_index])
        print("All elements were considered outliers. Keeping the largest one.")
    return new_area, new_perim, new_img

# Main script to process images and create CSV
if __name__ == "__main__":
    root_folder = "erthrocytesIDB\erthrocytesIDB"
    subfolders = ["erythrocytesIDB2", "erythrocytesIDB3"]
    image_types = ["mask-circular.jpg", "mask-elongated.jpg"]
    csv_filename = "cell_features.csv"

    with open(csv_filename, mode='w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["area", "perimeter", "circularity", "class"]) # Write header

        for subfolder in subfolders:
            subfolder_path = os.path.join(root_folder, subfolder)
            if not os.path.exists(subfolder_path) or not os.path.isdir(subfolder_path):
                print(f"Subfolder '{subfolder_path}' not found.")
                continue

            for image_folder in sorted(os.listdir(subfolder_path)): # Sort to process in numerical order
                image_folder_path = os.path.join(subfolder_path, image_folder)
                if not os.path.isdir(image_folder_path):
                    continue

                for image_type in image_types:
                    # Determine class based on image_type
                    if image_type == "mask-circular.jpg":
                        cell_class = 0 # 0 for healthy (circular)
                    elif image_type == "mask-elongated.jpg":
                        cell_class = 1 # 1 for elongated
                    else:
                        cell_class = -1 # Optional: Handle other image types if needed

                    image_path = os.path.join(image_folder_path, image_type)
                    if os.path.exists(image_path):
                        print(f"Processing image: {image_path}")
                        result, num_features = image_prep(image_path)
                        areaArray, perimArray = extract_area_perim(result, num_features)

                        if areaArray:
                            areaArray.pop(0) # Remove background area
                        if perimArray:
                            perimArray.pop(0) # Remove background perimeter
                        circularityArray = extract_circularity(areaArray, perimArray)
                        relativeAreaArray, relativePerimArray = convert_to_relative(areaArray, perimArray)
                        area, perimeter, img = removeOutliers(relativeAreaArray, relativePerimArray, result.copy()) # Use result.copy()

                        for i in range(len(area)):
                            if i < len(perimeter) and i < len(circularityArray):
                                csv_writer.writerow([area[i], perimeter[i], circularityArray[i], cell_class])

    print(f"CSV file '{csv_filename}' created successfully.")
