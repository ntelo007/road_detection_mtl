import rasterio
from rasterio.merge import merge
import glob
import os
import argparse
import numpy as np, sys
import matplotlib.pyplot as plt
import cv2
import shutil
import random
import geoTools as gT
from scipy.ndimage.morphology import *
import gdal
import geopandas as gpd
import fnmatch
import data_utils
from data_utils import affinity_utils
from numpy.lib.stride_tricks import as_strided


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
    # plt.hist(image_bands[0][0].flatten(), bins=bins, weights=image_bands[0][1].flatten(), label='red band')
    # plt.hist(image_bands[1][0].flatten(), bins=bins, weights=image_bands[1][1].flatten(), label='green band')
    # plt.hist(image_bands[2][0].flatten(), bins=bins, weights=image_bands[2][1].flatten(), label='blue band')
    # plt.show()

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
    # computes  min and max boundaries that enclose the 99% of the data
    image_histograms = compute_hist_for_single_image(image)
    bands = ['red', 'green', 'blue']
    boundaries = []
    for i in range(3):
        hist = image_histograms[0][i][0]
        sum_of_values = np.sum(image_histograms[0][i][0])
        cdf = np.around(np.cumsum(hist / sum_of_values), decimals=3) * 100

        bin_edges = image_histograms[0][i][1]

        min_index = np.where(cdf > 0.499)
        min_edge = cdf[min_index[0][0]]

        max_index_inv = np.where(np.flip(cdf) < 99.6)
        ind = int(len(cdf) - cdf[max_index_inv[0][0]] - 1)
        max_edge = bin_edges[ind]

        boundaries.append([min_edge, max_edge])

    return boundaries


def convert_16_to_8_bits(source, boundaries, destination, region):
    if type(source) == str:
        if source[len(source) - 3:] == 'tif':
            # collect the image bands
            with rasterio.open(source) as file:
                bands = []
                nodata = []
                for band in file.indexes:
                    image_band = file.read(band)
                    bands.append(image_band)
                    nodata.append(np.where(image_band == 0))

            # convert to 8 bit, boundaries contain global min and max values of every band
            red_band = 255.0 * ((bands[0] - boundaries[0][0]) / (boundaries[0][1] - boundaries[0][0]))
            green_band = 255.0 * ((bands[1] - boundaries[1][0]) / (boundaries[1][1] - boundaries[1][0]))
            blue_band = 255.0 * ((bands[2] - boundaries[2][0]) / (boundaries[2][1] - boundaries[2][0]))

            # initialization of a CLAHE or Contrast Limited Adaptive Histogram Equalization object
            clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))

            ## The default image size of Spacenet Dataset is 1300x1300.
            img_rgb = np.zeros((1300, 1300, 3), dtype=np.uint8)
            img_rgb[:, :, 0] = clahe.apply(np.asarray(red_band, dtype=np.uint8))
            img_rgb[:, :, 0][nodata[0]] = 0  # keep no data values as 0s
            img_rgb[:, :, 1] = clahe.apply(np.asarray(green_band, dtype=np.uint8))
            img_rgb[:, :, 1][nodata[1]] = 0
            img_rgb[:, :, 2] = clahe.apply(np.asarray(blue_band, dtype=np.uint8))
            img_rgb[:, :, 2][nodata[2]] = 0

            # save image to output dir
            name = source.split('\\')[-1]  # example: 'SN3_roads_train_AOI_2_Vegas_PS-RGB_img1.tif'
            actual_name = region + '_' + name.split('_')[-1].split('.tif')[0]  # example: 'img1'
            output_path = os.path.join(destination, actual_name + '.png')
            cv2.imwrite(output_path, img_rgb[:, :, ::-1])
        else:
            if not os.path.isdir(destination):
                os.makedirs(destination)
            # Conversion of images to 8 bit
            search_criterion = "*.tif"
            q = os.path.join(source, search_criterion)
            src_images = glob.glob(q)
            for image in src_images:
                convert_16_to_8_bits(image, boundaries, destination, region)


def find_unique_images(txt_file):
    """
    This function identifies all the unique image names that were the source files for the cropped images
    :param txt_file: txt file with all the names of the cropped images
    :return: a list containing all the unique image names
    """
    # Open the crops.txt file with read only permit
    f = open(txt_file, "r")
    # use readlines to read all lines in the file
    # The variable "lines" is a list containing all lines in the file
    lines = f.readlines()
    # close the file after reading the lines.
    f.close()

    unique_names = []
    for line in lines:
        words = line.split('_')
        if words[3] not in unique_names:
            unique_name = words[0] + "_" + words[1] + "_" + words[2] + "_" + words[3]
            unique_names.append(unique_name)
    f.close()
    return unique_names


def distribute_images(config, dirs):
    # find the number of unique images
    crops = dirs["crops_output"]
    # print("crops: ", crops)
    txt_file = os.path.join(crops, "crops.txt")
    # print("txt_file: ", txt_file)
    unique_img_names = find_unique_images(txt_file)
    # print("len(unique_img_names): ", len(unique_img_names))
    # print("unique_img_names: ", unique_img_names)

    # create source path variables
    images = os.path.join(crops, "images")
    centerline_gt = os.path.join(crops, "centerline_gt")
    gaussian_gt = os.path.join(crops, "gaussian_gt")
    two_road_gt = os.path.join(crops, "2m_road_gt")
    orientation_gt = os.path.join(crops, "orientation_gt")
    intersection_gt = os.path.join(crops, "intersection_gt")
    source_paths = [images, centerline_gt, gaussian_gt, two_road_gt, orientation_gt, intersection_gt]

    # create the destination folders
    train_dict = config["training_data"]["SpaceNet"]["train"]
    validation_dict = config["training_data"]["SpaceNet"]["validation"]
    test_dict = config["training_data"]["SpaceNet"]["test"]
    dataset_paths = [validation_dict, test_dict]

    # create destination directories if they don't exist
    if not os.path.exists(config["training_data"]["SpaceNet"]["base_dir"]):
        for i in dataset_paths:
            for key in list(i.values())[2:]:
                os.makedirs(key)

    # get percentages of each data set
    # train_perc = config['training_data']['SpaceNet']['train']['percentage'] / 100
    val_perc = config['training_data']['SpaceNet']['validation']['percentage'] / 100
    test_perc = config['training_data']['SpaceNet']['test']['percentage'] / 100
    number_of_imgs = len(unique_img_names)
    number_of_test_images = int(test_perc * number_of_imgs)
    number_of_val_images = int(val_perc * number_of_imgs)

    print("Total number of source images: ", number_of_imgs)
    print("Number of training images: ", (number_of_imgs - number_of_val_images - number_of_test_images))
    print("Number of validation images: ", number_of_val_images)
    print("Number of test images: ", number_of_test_images)

    # copy all files to the training destination folder
    def copytree(src, dst, symlinks=False, ignore=None):
        if not os.path.exists(dst):
            os.makedirs(dst)
        for item in os.listdir(src):
            s = os.path.join(src, item)
            d = os.path.join(dst, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, symlinks, ignore)
            else:
                shutil.copy2(s, d)

    copytree(images, train_dict["images"])
    copytree(centerline_gt, train_dict["centerline_gt"])
    copytree(gaussian_gt, train_dict["gaussian_gt"])
    copytree(two_road_gt, train_dict["2m_road_gt"])
    copytree(orientation_gt, train_dict["orientation_gt"])
    copytree(intersection_gt, train_dict["intersection_gt"])

    # distribute images according to their percentages
    count = 0

    while True:
        if count < number_of_val_images:
            # print("count of validation: ", count)
            random_image = random.choice(unique_img_names)
            # print("random_image: ", random_image)
            random_image_ = random_image + "_"
            # print("random_image_: ", random_image_)

            for file in os.listdir(train_dict["images"]):
                if fnmatch.fnmatch(file, "*{}*".format(random_image_)):
                    shutil.move(os.path.join(train_dict["images"], file), validation_dict["images"])
                    shutil.move(os.path.join(train_dict["centerline_gt"], file), validation_dict["centerline_gt"])
                    shutil.move(os.path.join(train_dict["gaussian_gt"], file), validation_dict["gaussian_gt"])
                    shutil.move(os.path.join(train_dict["2m_road_gt"], file), validation_dict["2m_road_gt"])
                    shutil.move(os.path.join(train_dict["orientation_gt"], file), validation_dict["orientation_gt"])
                    shutil.move(os.path.join(train_dict["intersection_gt"], file), validation_dict["intersection_gt"])
                    count += 1
            unique_img_names.remove(random_image)

        elif count < (number_of_val_images + number_of_test_images):
            # print("count of test: ", count)
            random_image = random.choice(unique_img_names)
            random_image_ = random_image + "_"

            for file in os.listdir(train_dict["images"]):
                if fnmatch.fnmatch(file, "*{}*".format(random_image_)):
                    shutil.move(os.path.join(train_dict["images"], file), test_dict["images"],
                                copy_function=shutil.copytree)
                    shutil.move(os.path.join(train_dict["centerline_gt"], file), test_dict["centerline_gt"],
                                copy_function=shutil.copytree)
                    shutil.move(os.path.join(train_dict["gaussian_gt"], file), test_dict["gaussian_gt"],
                                copy_function=shutil.copytree)
                    shutil.move(os.path.join(train_dict["2m_road_gt"], file), test_dict["2m_road_gt"],
                                copy_function=shutil.copytree)
                    shutil.move(os.path.join(train_dict["orientation_gt"], file), test_dict["orientation_gt"],
                                copy_function=shutil.copytree)
                    shutil.move(os.path.join(train_dict["intersection_gt"], file), test_dict["intersection_gt"],
                                copy_function=shutil.copytree)
                    count += 1

            # count = len([name for name in os.listdir(test_dict["intersection_gt"]) if os.path.isfile(os.path.join(test_dict["intersection_gt"], name))])

            unique_img_names.remove(random_image)

        else:
            # for name in unique_img_names:
            #     name = name + "_"
            #     for file in image_names:
            #         if fnmatch.fnmatch(file, "*{}*".format(name)):
            #             shutil.copy(os.path.join(images, file), train_dict["images"])
            #             shutil.copy(os.path.join(centerline_gt, file), train_dict["centerline_gt"])
            #             shutil.copy(os.path.join(gaussian_gt, file), train_dict["gaussian_gt"])
            #             shutil.copy(os.path.join(two_road_gt, file), train_dict["2m_road_gt"])
            #             shutil.copy(os.path.join(orientation_gt, file), train_dict["orientation_gt"])
            #             shutil.copy(os.path.join(intersection_gt, file), train_dict["intersection_gt"])
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


def neighbours(x, y, image):
    """Return 8-neighbours of image point P1(x,y), in a clockwise order"""
    x_1, y_1, x1, y1 = x - 1, y - 1, x + 1, y + 1
    return [image[x_1][y], image[x_1][y1], image[x][y1], image[x1][y1], image[x1][y], image[x1][y_1], image[x][y_1],
            image[x_1][y_1]]


def getSkeletonIntersection(skeleton):
    """ Given a skeletonised image, it will give the coordinates of the intersections of the skeleton.

    Keyword arguments:
    skeleton -- the skeletonised image to detect the intersections of

    Returns:
    List of 2-tuples (x,y) containing the intersection coordinates
    """
    # A biiiiiig list of valid intersections             2 3 4
    # These are in the format shown to the right         1 C 5
    #                                                    8 7 6
    validIntersection = [[0, 1, 0, 1, 0, 0, 1, 0], [0, 0, 1, 0, 1, 0, 0, 1], [1, 0, 0, 1, 0, 1, 0, 0],
                         [0, 1, 0, 0, 1, 0, 1, 0], [0, 0, 1, 0, 0, 1, 0, 1], [1, 0, 0, 1, 0, 0, 1, 0],
                         [0, 1, 0, 0, 1, 0, 0, 1], [1, 0, 1, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0, 1],
                         [0, 1, 0, 1, 0, 0, 0, 1], [0, 1, 0, 1, 0, 1, 0, 0], [0, 0, 0, 1, 0, 1, 0, 1],
                         [1, 0, 1, 0, 0, 0, 1, 0], [1, 0, 1, 0, 1, 0, 0, 0], [0, 0, 1, 0, 1, 0, 1, 0],
                         [1, 0, 0, 0, 1, 0, 1, 0], [1, 0, 0, 1, 1, 1, 0, 0], [0, 0, 1, 0, 0, 1, 1, 1],
                         [1, 1, 0, 0, 1, 0, 0, 1], [0, 1, 1, 1, 0, 0, 1, 0], [1, 0, 1, 1, 0, 0, 1, 0],
                         [1, 0, 1, 0, 0, 1, 1, 0], [1, 0, 1, 1, 0, 1, 1, 0], [0, 1, 1, 0, 1, 0, 1, 1],
                         [1, 1, 0, 1, 1, 0, 1, 0], [1, 1, 0, 0, 1, 0, 1, 0], [0, 1, 1, 0, 1, 0, 1, 0],
                         [0, 0, 1, 0, 1, 0, 1, 1], [1, 0, 0, 1, 1, 0, 1, 0], [1, 0, 1, 0, 1, 1, 0, 1],
                         [1, 0, 1, 0, 1, 1, 0, 0], [1, 0, 1, 0, 1, 0, 0, 1], [0, 1, 0, 0, 1, 0, 1, 1],
                         [0, 1, 1, 0, 1, 0, 0, 1], [1, 1, 0, 1, 0, 0, 1, 0], [0, 1, 0, 1, 1, 0, 1, 0],
                         [0, 0, 1, 0, 1, 1, 0, 1], [1, 0, 1, 0, 0, 1, 0, 1], [1, 0, 0, 1, 0, 1, 1, 0],
                         [1, 0, 1, 1, 0, 1, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0],
                         [0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 1],
                         [1, 0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0, 1, 0],
                         [1, 0, 0, 0, 0, 0, 1, 0]]

    if isinstance(skeleton, str):
        skeleton = cv2.imread(skeleton)
    image = skeleton[:, :] / 255.0
    intersections = []
    for x in range(1, len(image) - 1):
        for y in range(1, len(image[x]) - 1):
            # If we have a white pixel
            if image[x][y] == 1.0:
                nbours = neighbours(x, y, image)
                if nbours in validIntersection:
                    intersections.append((y, x))
    # Filter intersections to make sure we don't count them twice or ones that are very close together
    for point1 in intersections:
        for point2 in intersections:
            if (((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) < 10 ** 2) and (point1 != point2):
                intersections.remove(point2)
    # Remove duplicates
    intersections = list(set(intersections))
    return intersections


def find_neighbors_within_distance(image, x, y, d):
    height, width = image.shape
    neighbors = []

    # define search boundaries
    a_x = 0 if x-d < 0 else x-d
    b_x = 256 if x+d+1 > 255 else x+d+1

    a_y = 0 if y-d < 0 else y-d
    b_y = 256 if y+d+1 > 255 else y+d+1

    for n_x in range(a_x, b_x):
        for n_y in range(a_y, b_y):
            neighbors.append((n_x, n_y))

    return neighbors


def create_road_masks(tif_dir, road_dir, output_dir, region):
    # if the output folder doesn't exist, create it
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
        os.makedirs(os.path.join(output_dir, 'label_tif'))
        os.makedirs(os.path.join(output_dir, 'gaussian_gt'))
        os.makedirs(os.path.join(output_dir, 'centerline_gt'))
        os.makedirs(os.path.join(output_dir, '2m_road_gt'))
        os.makedirs(os.path.join(output_dir, 'orientation_gt'))
        os.makedirs(os.path.join(output_dir, 'intersection_gt'))

    # Create geojson file iterator
    search_criterion = "*.geojson"
    q = os.path.join(road_dir, search_criterion)
    src_roads = glob.glob(q)

    # The default image size of Spacenet 3 Dataset is 1300x1300.
    black_image = np.zeros((1300, 1300), dtype=np.uint8)
    failure_count = 0
    index = 0

    for geojson_file in src_roads:
        index += 1
        # get the name of the road file that matches the RGB-Pan tif
        name = geojson_file.split('\\')[-1]  # example: 'SN3_roads_train_AOI_2_Vegas_geojson_roads_img1.geojson'
        actual_name = name.split('_')[-1].split('.geojson')[0]  # example: 'img1'
        final_name = region + '_' + name.split('_')[-1].split('.geojson')[0]  # example: 'img1'
        # output_road_mask_file = os.path.join(output_dir, actual_name + '.png') # example: 'C:\\...\\road_masks\\img1.png'
        out_tif_file = os.path.join(output_dir, 'label_tif', final_name + '.tif')
        out_png_file = os.path.join(output_dir, 'gaussian_gt', final_name + '.png')
        out_centerline_png_file = os.path.join(output_dir, 'centerline_gt', final_name + '.png')
        out_2m_png_file = os.path.join(output_dir, '2m_road_gt', final_name + '.png')
        out_orientation_png_file = os.path.join(output_dir, 'orientation_gt', final_name + '.png')
        out_intersection_png_file = os.path.join(output_dir, 'intersection_gt', final_name + '.png')
        for tif in os.listdir(tif_dir):
            if actual_name in tif:
                tif_file = os.path.join(tif_dir, tif)
                break

        status = gT.ConvertToRoadSegmentation(tif_file, geojson_file, out_tif_file)

        if status != 0:
            print("|xxx-> Not able to convert the file {}. <-xxx".format(name))
            failure_count += 1
            cv2.imwrite(out_png_file, black_image)
        else:
            gt_dataset = gdal.Open(out_tif_file, gdal.GA_ReadOnly)
            if not gt_dataset:
                continue
            gt_array = gt_dataset.GetRasterBand(1).ReadAsArray()

            # creation of gaussian masks
            distance_array = distance_transform_edt(1 - (gt_array / 255))
            std = 15
            distance_array = np.exp(-0.5 * (distance_array * distance_array) / (std * std))
            distance_array *= 255

            cv2.imwrite(out_png_file, distance_array)

            # creation of centerline gt masks
            cv2.imwrite(out_centerline_png_file, gt_array)

            # creation of intersection gt masks --> distance == 4
            height, width = gt_array.shape
            new_image = np.zeros((height, width))
            intersections = getSkeletonIntersection(gt_array)

            # # find boundary points
            # boundary_points = []
            # a_side = np.where(gt_array[:, 0] == 255)
            # b_side = np.where(gt_array[:, 1299] == 255)
            # c_side = np.where(gt_array[0, :] == 255)
            # d_side = np.where(gt_array[1299, :] == 255)
            #
            # if len(a_side[0]) > 0:
            #     for i in a_side[0]:
            #         a_point = [i, 0]
            #         boundary_points.append(a_point)
            # if len(b_side[0]) > 0:
            #     for i in b_side[0]:
            #         b_point = [i, 1299]
            #         boundary_points.append(b_point)
            # if len(c_side[0]) > 0:
            #     for i in c_side[0]:
            #         c_point = [0, i]
            #         boundary_points.append(c_point)
            # if len(d_side[0]) > 0:
            #     for i in d_side[0]:
            #         d_point = [1299, i]
            #         boundary_points.append(d_point)


            for point in intersections:
                nbours = find_neighbors_within_distance(new_image, point[0], point[1], 6)
                for nbour in nbours:
                    new_image[nbour[1], nbour[0]] = 255
            # for b in boundary_points:
            #     bps = find_neighbors_within_distance(new_image, b[0], b[1], 6)
            #     for new_b in bps:
            #         new_image[new_b[0], new_b[1]] = 255
            cv2.imwrite(out_intersection_png_file, new_image)

            # creation of 2m width gt masks
            struct2 = generate_binary_structure(2, 2)
            buff_array = binary_dilation(gt_array, structure=struct2, iterations=3).astype(gt_array.dtype)
            buff_array = np.where(buff_array == 1, 255, 0)
            cv2.imwrite(out_2m_png_file, buff_array)

            # creation of orientation masks
            gaussian_road_mask = distance_array.astype(np.float)
            # Smooth the road graph with tolerance = 4 and get keypoints of road segments
            keypoints = affinity_utils.getKeypoints(gaussian_road_mask, smooth_dist=4)
            h, w = gaussian_road_mask.shape
            # generate orienation mask in euclidean and polar domain
            vecmap_euclidean, orienation_angles = affinity_utils.getVectorMapsAngles((h, w), keypoints, theta=10,
                                                                                     bin_size=10)
            cv2.imwrite(out_orientation_png_file, orienation_angles)

    # shutil.rmtree(os.path.join(output_dir, 'label_tif'))
    print("Not able to convert {} files.".format(failure_count))


def verify_image(img_file):
    try:
        img = cv2.imread(img_file)
    except:
        return False
    return True


def create_crops(output_crops_path, uncropped_masks, uncropped_imgs, size, overlap, image_suffix, gt_suffix):
    # if the crops path doesn't exist, create it, along with the sub-folders
    if not os.path.exists(output_crops_path):
        os.makedirs(output_crops_path)
        os.makedirs(output_crops_path + "/images")
        os.makedirs(output_crops_path + "/centerline_gt")
        os.makedirs(output_crops_path + "/2m_road_gt")
        os.makedirs(output_crops_path + "/gaussian_gt")
        os.makedirs(output_crops_path + "/orientation_gt")
        os.makedirs(output_crops_path + "/intersection_gt")

    # create a txt file with the names of the cropped road masks
    text_file_path = os.path.join(output_crops_path, 'crops.txt')
    crops_text_file = open(text_file_path, 'w')

    # keep a list of images with problems
    failure_images = []

    # Create source gt mask file iterator
    search_criterion = "*.png"
    uncropped_centerl_gt = os.path.join(uncropped_masks, "centerline_gt")
    q = os.path.join(uncropped_centerl_gt, search_criterion)
    src_centerl_gt_img = glob.glob(q)

    for gt_file_path in src_centerl_gt_img:
        path, name = os.path.split(gt_file_path)
        name = name.split('.')[0]

        # create full path variables of gt and img files
        img_file_path = os.path.join(uncropped_imgs, name + image_suffix)
        two_m_road_gt_path = uncropped_masks + "\\2m_road_gt\\" + name + image_suffix
        gaussian_gt_path = uncropped_masks + "\\gaussian_gt\\" + name + image_suffix
        orientation_gt_path = uncropped_masks + "\\orientation_gt\\" + name + image_suffix
        intersection_gt_path = uncropped_masks + "\\intersection_gt\\" + name + image_suffix

        # if this image cannot be found, then append it to failures
        if not verify_image(img_file_path):
            failure_images.append(gt_file_path)
            continue

        # open images
        image = cv2.imread(img_file_path)
        two_m_road_gt = cv2.imread(two_m_road_gt_path, 0)
        gaussian_gt = cv2.imread(gaussian_gt_path, 0)
        centerline_gt = cv2.imread(gt_file_path, 0)
        orientation_gt = cv2.imread(orientation_gt_path, 0)
        intersection_gt = cv2.imread(intersection_gt_path, 0)

        # similar check
        if image is None:
            failure_images.append(img_file_path)
            continue

        if centerline_gt is None:
            failure_images.append(img_file_path)
            continue

        # get information about height, width and nu. of channels for the image
        H, W, C = image.shape
        maxx = (H - size) / overlap
        maxy = (W - size) / overlap
        image_pixels_count = 256 * 256 * 3

        for x in range(int(maxx) + 1):
            for y in range(int(maxy) + 1):
                im_ = image[x * overlap:x * overlap + size, y * overlap:y * overlap + size, :]
                if np.sum(im_ == 0) < (0.25 * image_pixels_count):
                    centerline_gt_ = centerline_gt[x * overlap:x * overlap + size, y * overlap:y * overlap + size]
                    gaussian_gt_ = gaussian_gt[x * overlap:x * overlap + size, y * overlap:y * overlap + size]
                    two_m_road_gt_ = two_m_road_gt[x * overlap:x * overlap + size, y * overlap:y * overlap + size]
                    orientation_gt_ = orientation_gt[x * overlap:x * overlap + size, y * overlap:y * overlap + size]
                    intersection_gt_ = intersection_gt[x * overlap:x * overlap + size, y * overlap:y * overlap + size]

                    ############################
                    # New attempt
                    #############################
                    new_image = np.zeros((256, 256))
                    intersections = getSkeletonIntersection(centerline_gt_)

                    # find boundary points
                    boundary_points = []

                    a_side = np.where(centerline_gt_[:, 0] == 255)
                    b_side = np.where(centerline_gt_[:, 255] == 255)
                    c_side = np.where(centerline_gt_[0, :] == 255)
                    d_side = np.where(centerline_gt_[255, :] == 255)

                    if len(a_side[0]) > 0:
                        for i in a_side[0]:
                            a_point = [i, 0]
                            boundary_points.append(a_point)
                    if len(b_side[0]) > 0:
                        for i in b_side[0]:
                            b_point = [i, 255]
                            boundary_points.append(b_point)
                    if len(c_side[0]) > 0:
                        for i in c_side[0]:
                            c_point = [0, i]
                            boundary_points.append(c_point)
                    if len(d_side[0]) > 0:
                        for i in d_side[0]:
                            d_point = [255, i]
                            boundary_points.append(d_point)

                    for point in intersections:
                        nbours = find_neighbors_within_distance(new_image, point[0], point[1], 6)
                        for nbour in nbours:
                            new_image[nbour[1], nbour[0]] = 255
                    for b in boundary_points:
                        bps = find_neighbors_within_distance(new_image, b[1], b[0], 6)
                        for new_b in bps:
                            new_image[new_b[1], new_b[0]] = 255

                    crops_text_file.write('{}_{}_{}\n'.format(name, x, y))
                    cv2.imwrite(output_crops_path + '/images/{}_{}_{}{}'.format(name, x, y, image_suffix), im_)
                    cv2.imwrite(output_crops_path + '/centerline_gt/{}_{}_{}{}'.format(name, x, y, gt_suffix),
                                centerline_gt_)
                    cv2.imwrite(output_crops_path + '/gaussian_gt/{}_{}_{}{}'.format(name, x, y, gt_suffix),
                                gaussian_gt_)
                    cv2.imwrite(output_crops_path + '/2m_road_gt/{}_{}_{}{}'.format(name, x, y, gt_suffix),
                                two_m_road_gt_)
                    cv2.imwrite(output_crops_path + '/orientation_gt/{}_{}_{}{}'.format(name, x, y, gt_suffix),
                                orientation_gt_)
                    cv2.imwrite(output_crops_path + '/intersection_gt/{}_{}_{}{}'.format(name, x, y, gt_suffix),
                                new_image)
                else:
                    failure_images.append(str('{}_{}_{}\n'.format(name, x, y)))

    crops_text_file.close()

    if len(failure_images) > 0:
        print("Unable to crop {} images. Their names are:\n{}".format(len(failure_images), failure_images))


def convert_to_binary(image, threshold):
    """
    This function converts an input image to binary according to a user-defined threshold
    :param threshold:
    :param image: input image
    :return: binary image
    """
    binary = np.copy(image / 255.0)
    binary[image > threshold] = 1
    binary[image <= 0.5] = 0
    return binary
