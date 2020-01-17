"""
This script computes the histogram of an image
"""

import numpy as np
import rasterio
import argparse
import os
import glob


def compute_hist_for_single_image(image, bins='auto'):
    with rasterio.open(image) as file:
        image_bands = []
        for band in file.indexes:
            image_band = file.read(band)
            image_bands.append(image_band)
    histograms = []
    for i in image_bands:
        hist, bin_edges = np.histogram(i, bins=bins)
        histograms.append([hist, bin_edges])

    return histograms


def compute_hist_of_multiple_images(input_dir):

    if not os.path.isdir(input_dir):
        print("You typed: {0}").format(input_dir)
        print("This directory does not exist. The computation of histograms is cancelled.")
        return None

    else:
        # a search criterion, that every file should have the ending .tif
        search_criterion = "*.tif"
        q = os.path.join(input_dir, search_criterion)
        # creates a list of all images within the input dir that have the ending .tif
        src_images = glob.glob(q)
        # creates a list to store every histogram from every image
        list_of_histograms = []
        for i in src_images:
            h = compute_hist_for_single_image(i)
            list_of_histograms.append(h)
        print("Example histogram")
        print(list_of_histograms[0][0])
        return list_of_histograms

def main():
    # Create a parser object and request info from the user. Store it in the variable args.
    parser = argparse.ArgumentParser()
    # ask from the user to provide the full path of an image
    parser.add_argument('--image_path',
                        '-i',
                        help='Please provide the full path of the image. ',
                        type=str,
                        required=False)
    parser.add_argument('--input_path',
                        '-p',
                        help='Please provide the full path of input directory. ',
                        type=str,
                        required=False)
    args = parser.parse_args()
    compute_hist_of_multiple_images(args.input_path)
    print("The process is finished.")


if __name__ == "__main__":
# execute only if run as a script
    main()