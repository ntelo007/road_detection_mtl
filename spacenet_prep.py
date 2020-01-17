"""
The SpaceNet imagery is distributed as 16-bit imagery, 
and the road network is distributed as a line-string vector GeoJSON format
This script converts the images to 8-bit and creates the road masks
"""

import argparse
import time
import os
import numpy as np
import tifffile as tif


def convert_to_8bit(input_dir, output_dir):
    if not os.path.isdir(input_dir):
        print("You typed: {0}").format(input_dir))
        print("This directory does not exist. The script is terminating")

    else:
        # if the output dir doesn't exist, create it
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for image_16_bits in os.listdir(input_dir):
            if image_16_bits.endswith(".tif"):
                image_name = image_16_bits.split('/')[-1].replace('.tif', '.png')

                # read image and collect R,G,B bands
                img = tif.imread(image_16_bits)
                red_band = np.asarray(image_16_bits[:,:,0], dtype=np.float)
                green_band = np.asarray(image_16_bits[:, :, 0], dtype=np.float)
                blue_band = np.asarray(image_16_bits[:, :, 0], dtype=np.float)

                




def main():
    # Create a parser object, request info from the user and store it in the variable args
    parser = argparse.ArgumentParser()
    # ask from the user to provide the full path of the input directory of the 16 bit images
    parser.add_argument('--input_dir',
                        '-i',
                        help='Please provide the input directory (RGB-Pansharpened image folder).',
                        type=str,
                        required=True)
    # ask from the user to provide the full path of the output directory where the 8 bit images will be stored
    parser.add_argument('--output_dir',
                        '-o',
                        help='Please provide the output directory.',
                        type=str,
                        required=True)
    args = parser.parse_args()

    start = time.clock()
    convert_to_8bit(args.input_dir, args.output_dir)
    end = time.clock()

    print("The conversion of the input 16-bit images to 8-bit images is finished. The procedure lasted {1}s.".format(end - start))