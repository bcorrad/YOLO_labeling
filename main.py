from skimage.measure import find_contours
from PIL import Image
import numpy as np
from shapely.geometry import Polygon
import cv2
import matplotlib.pyplot as plt
from descartes import PolygonPatch
import os
import argparse


def show_polygons(image, polygons, show=True):
    fig, ax = plt.subplots()
    plt.imshow(np.zeros_like(image))
    polygons_list = []
    for poly in polygons:
        exterior = np.array(poly.exterior.coords)
        exterior_norm = []
        # flip x and y coordinates and normalize for each point of the polygon
        for e in exterior:
            # swap coordinates
            e[[0, 1]] = e[[1, 0]]
            # normalize coordinates on image shape
            x = e[0]/image.shape[0]
            y = e[1]/image.shape[1]
            exterior_norm.append([x, y])
        polygons_list.append(exterior_norm)
        flipped_poly = Polygon(exterior)
        ax.add_patch(PolygonPatch(flipped_poly, fc='blue', ec='white', alpha=0.9))
    if show:
        plt.imshow(image)
        plt.show()
    return polygons_list


def save_polygon_coordinates(poly_list, filepath, class_idx=0):
    for list_of_lists in poly_list:
        with open(filepath, 'a+') as f:
            points = ""
            for lst in list_of_lists:
                points += " ".join([str(c) for c in lst])+" "
            f.write(str(class_idx)+" "+points+"\n")


def segmentation_to_polygons(image):
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    # Find contours of all white regions
    contours = find_contours(gray, 0)
    # Create a list to store the polygons
    polygons = []
    # Iterate over each contour
    for contour in contours:
        # Create a polygon from the contour
        try:
            poly = Polygon(contour)
            polygons.append(poly)
        except Exception as e:
            print(e)
    return polygons


def process_folder(folder_path, labels_dir_path, image_extensions):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if any(file.endswith(ext) for ext in image_extensions):
                image_path = os.path.join(root, file)
                txt_filename = f'{os.path.splitext(file)[0]}.txt'
                with Image.open(image_path) as im:
                    image = np.array(im)
                polygons = segmentation_to_polygons(image)
                flipped_poly = show_polygons(image, polygons)
                save_polygon_coordinates(flipped_poly, os.path.join(labels_dir_path, txt_filename))
                print(f'Saved {image_path} --> {txt_filename}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--images-dir-path', nargs='?', type=str, default='./test_inclusi_maschere', help='folder containing segmentation maps')
    parser.add_argument('--labels-dir-path', nargs='?', type=str, default='./test_inclusi_labels', help='folder to save labels')

    args = parser.parse_args()
    images_dir_path = args.images_dir_path
    labels_dir_path = args.labels_dir_path
    image_extensions = ['.jpg', '.png']
    process_folder(images_dir_path, labels_dir_path, image_extensions)
