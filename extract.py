import os
import csv
import numpy as np
import cv2
from skimage.feature import canny
from scipy import ndimage as ndi
import math




def image_prep(path):
    mask_size = 400
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    edges = canny(img/255.)
    fill_img = ndi.binary_fill_holes(edges)
    label_objects, nb_labels = ndi.label(fill_img)
    sizes = np.bincount(label_objects.ravel())
    mask_sizes = sizes > mask_size
    mask_sizes[0] = 0
    img_cleaned = mask_sizes[label_objects]
    labeled_img, num_features = ndi.label(img_cleaned)
    return labeled_img, num_features

def numofneighbour(mat, i, j, searchValue):
    count = 0;

    # UP
    if (i > 0 and mat[i - 1][j] == searchValue):
        count += 1;

        # LEFT
    if (j > 0 and mat[i][j - 1] == searchValue):
        count += 1;

        # DOWN
    if (i < len(mat) - 1 and mat[i + 1][j] == searchValue):
        count += 1

    # RIGHT
    # RIGHT
    if (j < len(mat[i]) - 1):
        count += 1

    return count;


# Returns sum of perimeter of shapes formed with 1s
def findperimeter(mat, num_features):
    perimeter = [0] * (num_features + 1)  # Adjusted the size of the perimeter list

    # Traversing the matrix and finding ones to calculate their contribution.
    for i in range(len(mat)):
        for j in range(0, len(mat[i])):
            if mat[i][j] != 0:
                # Calculate the perimeter contribution based on the current feature
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
            circularity.append(0)  # Handling division by zero

    return circularity

def convert_to_relative(cellAreas, cellPerims):
    if not cellAreas or not cellPerims:
        return cellAreas, cellPerims

    relativeArea = []
    relativePerim = []

    if cellAreas:
        averageCellSize = sum(cellAreas) / len(cellAreas)
        relativeArea = [area / averageCellSize for area in cellAreas]

    if cellPerims:
        averagePerimSize = sum(cellPerims) / len(cellPerims)
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
    new_img = img.copy()  # Make a copy to avoid modifying the original image

    for i in range(len(area)):
        if area[i] <= mean + 3.5 * std:
            new_area.append(area[i])
            new_perim.append(perim[i])
        else:
            print("Popping feature:", i + 1)  # Adjust index for printing the correct feature index
            new_img = removeFromImg(new_img, i + 1)  # Adjust index for removing the correct feature

    # Check if at least one element remains after outlier removal
    if len(new_area) == 0:
        # If all elements were considered outliers, keep the largest one
        max_area_index = np.argmax(area)
        new_area.append(area[max_area_index])
        new_perim.append(perim[max_area_index])
        print("All elements were considered outliers. Keeping the largest one.")

    return new_area, new_perim, new_img

#......................................................................................................

main_folder = "."

# Define the output text file name
output_file = "healthyCells.txt"

# Open the text file in write mode
with open(output_file, 'w') as txtfile:
    # Write the data to the text file in the specified format
    txtfile.write("[\n")

    # Iterate through each subfolder
    for subfolder in ['erythrocytesIDB2', 'erythrocytesIDB3']:
        subfolder_path = os.path.join(main_folder, subfolder)

        # Check if the subfolder path exists
        if not os.path.exists(subfolder_path):
            print(f"Folder '{subfolder}' not found.")
            continue

        # Iterate through each sub-subfolder
        for sub_subfolder in os.listdir(subfolder_path):
            sub_subfolder_path = os.path.join(subfolder_path, sub_subfolder)

            # Check if the path is a directory
            if os.path.isdir(sub_subfolder_path):
                # Construct the path to the image file
                image_path = os.path.join(sub_subfolder_path, "mask-circular.jpg")

                # Check if the image file exists
                if not os.path.exists(image_path):
                    print(f"Image file not found in '{sub_subfolder_path}'. Skipping...")
                    continue

                # Perform image processing
                result, num_features = image_prep(image_path)
                areaArray, perimArray = extract_area_perim(result, num_features)
                if areaArray:
                    areaArray.pop(0)
                if perimArray:
                    perimArray.pop(0)
                circularityArray = extract_circularity(areaArray, perimArray)
                relativeAreaArray, relativePerimArray = convert_to_relative(areaArray, perimArray)
                imag = removeFromImg(result, num_features)
                area, perimeter, img = removeOutliers(relativeAreaArray, relativePerimArray, imag)

                # Write the data to the text file
                for i in range(len(area)):
                    if i == len(area) - 1 and subfolder == 'erythrocytesIDB3' and sub_subfolder == 'image_120':
                        # If it's the last data point, omit the comma and newline character
                        txtfile.write(f"[{area[i]}, {perimeter[i]}, {circularityArray[i]}]\n")
                    else:
                        txtfile.write(f"[{area[i]}, {perimeter[i]}, {circularityArray[i]}],\n")

    txtfile.write("]\n")

print("Data extraction and writing to text file completed successfully.")