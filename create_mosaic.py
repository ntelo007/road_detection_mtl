"""
This script creates a raster mosaic.
The user provides the full path of a directory(ies) that contains the input images and
the full path of the output directory where the mosaic will be stored.
The user can change the metadata (no data value, transform, etc.) of the mosaic output.
"""

import rasterio
from rasterio.merge import merge
import glob
import os
import argparse


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


def main():
    # Create a parser object and request info from the user. Store it in the variable args.
    parser = argparse.ArgumentParser()
    # ask from the user to provide the full path of the input directory of the 16 bit images
    parser.add_argument('--input_dir',
                        '-i',
                        help='Please provide the input directory. ',
                        type=str,
                        required=True)
    # ask from the user to provide the full path of the output directory where the 8 bit images will be stored
    parser.add_argument('--output_dir',
                        '-o',
                        help='Please provide the output directory. ',
                        type=str,
                        required=True)
    parser.add_argument('--filename',
                        '-f',
                        help='Please provide the output filename without the ending (e.g. without .tif. ',
                        type=str,
                        required=True
                        )
    parser.add_argument('--secondary_input_dir',
                        '-s',
                        help='If desired, provide the secondary input directory',
                        type=str,
                        required=False)
    args = parser.parse_args()
    mosaic, meta = create_mosaic(args.input_dir,   args.secondary_input_dir)
    save_tif(mosaic, meta, args.output_dir, args.filename)
    print("The creation of the mosaic is finished.")


if __name__ == "__main__":
# execute only if run as a script
    main()




