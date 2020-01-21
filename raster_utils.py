import rasterio
from rasterio.merge import merge
import glob
import os
import argparse
import numpy as np


def compute_hist_for_single_image(image, bins='auto', range=None):

    if image is str:
        # read image and save its bands to a list
        with rasterio.open(image) as file:
            image_bands = []
            for band in file.indexes:
                image_band = file.read(band)
                image_bands.append(image_band)
        # store the histogram and bin edges of each image band
        histograms = []
        for i in image_bands:
            hist, bin_edges = np.histogram(i, bins=bins, range=range)
            histograms.append([hist, bin_edges])
    else:
        histograms = []
        for i in image:
            hist, bin_edges = np.histogram(i, bins=bins, range=range)
            histograms.append([hist, bin_edges])

    return histograms


def compute_hist_of_multiple_images(input_dir):
    """
    Creates a list of histograms for every band of every image withing a directory
    :param input_dir: Full path directory that contains images
    :return: list of histograms and bin edges of every image band of every image withing a directory
    """

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
        return list_of_histograms


def create_mosaic(input_dir, secondary_dir=None):
    """
    :param input_dir: Full path of the input directory holding images to be merged
    :param secondary_dir: Full path of the secondary directory holding images to be merged
    :return: the mosaic file and its metadata
    """
    if not os.path.isdir(input_dir):
        print("You typed: {0}").format(input_dir)
        print("This directory does not exist. The script is terminating")
    else:
        # a search criterion, that every file should have the ending .tif
        search_criterion = "*.tif"
        q = os.path.join(input_dir, search_criterion)

        # creates a list of all images within the input dir that have the ending .tif
        src_images = glob.glob(q)

        # creates an empty list for the images that will be part of the mosaic
        src_images_to_mosaic = []

        # reads images and adds them to the list
        for image in src_images:
            src = rasterio.open(image)
            src_images_to_mosaic.append(src)

        # if a second directory needs to be included in the mosaic, the images will be collected now
        if secondary_dir is not None:
            if not os.path.isdir(input_dir):
                print("You typed: {0}").format(input_dir)
                print("This directory does not exist. The script is terminating")
            else:
                q2 = os.path.join(secondary_dir, search_criterion)
                secondary_src_images = glob.glob(q2)
                # reads images and adds them to the list
                for image2 in secondary_src_images:
                    src2 = rasterio.open(image2)
                    src_images_to_mosaic.append(src2)

        # Merge all the images collected in the list
        mosaic, out_trans = merge(src_images_to_mosaic)

        # copy metadata of source images
        out_meta = src.meta.copy()

        # update the metadata
        out_meta.update({"driver": "GTiff",
                         "height": mosaic.shape[1],
                         "width": mosaic.shape[2],
                         "transform": out_trans,
                         "nodata": 0
                         })

        return mosaic, out_meta


def save_tif(image, meta, output_dir, filename):
    # Writes the raster to the desired output + filename
    image_name = output_dir + "\\" + filename + ".tif"
    with rasterio.open(image_name, "w", **meta) as dest:
        dest.write(image)


def equal_height_bins_hist(x, nbin):
    npt = len(x)
    return np.interp(np.linspace(0, npt, nbin + 1),
                     np.arange(npt),
                     np.sort(x))







