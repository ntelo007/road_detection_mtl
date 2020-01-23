import rasterio
from rasterio.merge import merge
import glob
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2
import shutil
import random


def compute_hist_for_single_image(image, bins=500):
    if type(image) == str:
        # read image and save its bands to a list
        with rasterio.open(image) as file:
            image_bands = []
            for band in file.indexes:
                image_band = file.read(band)
                mask = np.ma.masked_array(image_band, image_band == 0)
                mask = np.where(mask == True, 0, mask)
                mask = np.where(mask != 0, 1, mask)
                image_bands.append([image_band, mask])  # every band is accompanied with a nodata value mask
    else:
        image_bands = []
        for band in image:
            mask = np.ma.masked_array(band, band == 0)
            mask = np.where(mask == True, 0, mask)
            mask = np.where(mask != 0, 1, mask)
            image_bands.append([band, mask])
    # store the histogram and bin edges of each image band
    histograms = []
    red_hist, red_bounds = np.histogram(image_bands[0][0].flatten(), bins=bins, weights=image_bands[0][1].flatten())
    green_hist, green_bounds = np.histogram(image_bands[1][0].flatten(), bins=bins, weights=image_bands[1][1].flatten())
    blue_hist, blue_bounds = np.histogram(image_bands[2][0].flatten(), bins=bins, weights=image_bands[2][1].flatten())

    histograms.append([[red_hist, red_bounds], [green_hist, green_bounds], [blue_hist, blue_bounds]])

    # plot histograms for every image band
    #plt.hist(image_bands[0][0].flatten(), bins=bins, weights=image_bands[0][1].flatten(), label='red band')
    #plt.hist(image_bands[1][0].flatten(), bins=bins, weights=image_bands[1][1].flatten(), label='green band')
    #plt.hist(image_bands[2][0].flatten(), bins=bins, weights=image_bands[2][1].flatten(), label='blue band')
    #plt.show()

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


def compute_min_max_from_cdf(image):
    image_histograms = compute_hist_for_single_image(image)
    bands = ['red', 'green', 'blue']
    boundaries = []
    for i in range(3):
        hist = image_histograms[0][i][0]
        sum_of_values = np.sum(image_histograms[0][i][0])
        cdf = np.around(np.cumsum(hist/sum_of_values), decimals=3)*100

       # find the min and max boundaries that enclose the 99% of the data
        bin_edges = image_histograms[0][i][1]

        min_index = np.where(cdf > 0.499)
        min_edge = cdf[min_index[0][0]]

        max_index_inv = np.where(np.flip(cdf) < 99.6)
        ind = int(len(cdf) - cdf[max_index_inv[0][0]] - 1)
        max_edge = bin_edges[ind]

        boundaries.append([min_edge, max_edge])

    return boundaries


def convert_16_to_8_bits(source, boundaries, config):
    if type(source) == str:
        if source[len(source)-3:] == 'tif':
            # collect the image bands
            with rasterio.open(source) as file:
                bands = []
                for band in file.indexes:
                    image_band = file.read(band)
                    bands.append(image_band)
            # convert to 8 bit, boundaries contain global min and max values of every band
            red_band = 255.0 * ((bands[0] - boundaries[0][0]) / (boundaries[0][1] - boundaries[0][0]))
            green_band = 255.0 * ((bands[1] - boundaries[1][0]) / (boundaries[1][1] - boundaries[1][0]))
            blue_band = 255.0 * ((bands[2] - boundaries[2][0]) / (boundaries[2][1] - boundaries[2][0]))

            # initialization of a CLAHE or Contrast Limited Adaptive Histogram Equalization object
            clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))

            ## The default image size of Spacenet Dataset is 1300x1300.
            img_rgb = np.zeros((1300, 1300, 3), dtype=np.uint8)
            img_rgb[:, :, 0] = clahe.apply(np.asarray(red_band, dtype=np.uint8))
            img_rgb[:, :, 1] = clahe.apply(np.asarray(green_band, dtype=np.uint8))
            img_rgb[:, :, 2] = clahe.apply(np.asarray(blue_band, dtype=np.uint8))

            # save image to output dir
            output_dir = config['training_data']['SpaceNet']['train']['images']
            basename = os.path.basename(source)
            filename = basename[:(len(basename)-4)] + '.png'
            cv2.imwrite(os.path.join(output_dir, filename), img_rgb[:, :, ::-1])
        else:
            # Conversion of images to 8 bit
            search_criterion = "*.tif"
            q = os.path.join(source, search_criterion)
            src_images = glob.glob(q)
            for image in src_images:
                convert_16_to_8_bits(image, boundaries, config)


def distribute_images(config, type_of_image='gt'):
    search_criterion = "*.png"
    source = config['training_data']['SpaceNet']['train'][type_of_image]
    val_output = config['training_data']['SpaceNet']['validation'][type_of_image]
    test_output = config['training_data']['SpaceNet']['test'][type_of_image]

    q = os.path.join(source, search_criterion)
    src_images = glob.glob(q)
    number_of_initial_training_samples = len(src_images)
    percentage = config['training_data']['SpaceNet']['validation']['percentage']/100
    number_of_val_or_test_samples = int(percentage * number_of_initial_training_samples)

    count = 0
    while True:

        if count < number_of_val_or_test_samples:
            random_image = random.choice(src_images)
            shutil.move(random_image, val_output)
            count += 1
            src_images.remove(random_image)

        elif count < (2*number_of_val_or_test_samples):
            random_image = random.choice(src_images)
            shutil.move(random_image, test_output)
            count += 1
            src_images.remove(random_image)

        else:
            break


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
    # my own save tif function
    # Writes the raster to the desired output + filename
    image_name = output_dir + "\\" + filename + ".tif"
    with rasterio.open(image_name, "w", **meta) as dest:
        dest.write(image)


# not used yet
def equal_height_bins_hist(x, nbin):
    npt = len(x)
    return np.interp(np.linspace(0, npt, nbin + 1),
                     np.arange(npt),
                     np.sort(x))


def collect_inaccurate_raster_images(input_dir, raster_ending):
    # this function creates a list with all the rasters containing no data values

    # a search criterion, that every file should have the ending .tif
    search_criterion = "*." + raster_ending
    q = os.path.join(input_dir, search_criterion)
    # creates a list of all images within the input dir that have the ending provided by the user
    src_images = glob.glob(q)
    # creates an empty list for the images that will be part of the mosaic
    src_images_with_no_data_values = []

    # reads images and adds them to the list if no data value appeared in them
    for image in src_images:
        src = rasterio.open(image)
        no_data_value = src.nodata
        if no_data_value in src:
            src_images_with_no_data_values.append(src)

    return src_images_with_no_data_values


