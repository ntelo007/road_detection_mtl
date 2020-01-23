"""
The SpaceNet imagery is distributed as 16-bit imagery, 
and the road network is distributed as a line-string vector GeoJSON format.
This script converts the SpaceNet imagery to 8-bit, creates the road masks and distributes all files to final folders.
"""

import raster_utils
import json
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from time import perf_counter


def main():
    # Start the stopwatch / counter
    t1_start = perf_counter()

    with open('config.json') as json_file:

        # store config.json file as a dictionary
        config = json.load(json_file)

        # iterate through every region and convert images to 8-bit
        for region, dirs in config['pre-processing']['SpaceNet']['regions'].items():
            print('-'*80)
            print('Preparing region ', region, '...')
            keys = list(dirs)
            # creates a raster mosaic of all the satellite images withing the test and train directories
            mosaic, _ = raster_utils.create_mosaic(dirs[keys[0]], dirs[keys[1]])    #the second dir might be useless -> no gt exists
            # computes global histogram boundaries that contain 99% of the data
            boundaries = raster_utils.compute_min_max_from_cdf(mosaic)
            # convert images to 8 bit and save them all in the train folder of SpaceNet
            raster_utils.convert_16_to_8_bits(dirs[keys[0]], boundaries, config)
            # creation of road masks has to be added

            # cropping of both masks and images has to be added


            # distribute images from the train folders to test and val randomly
            raster_utils.distribute_images(config, 'images') # the second argument could be 'gt'
            print('Finished with region ', region)

    # Stop the stopwatch / counter
    t1_stop = perf_counter()

    print('SpaceNet dataset prep is done! Time elapsed: {0}s'.format(t1_stop-t1_start))


if __name__ == "__main__":
# execute only if run as a script
    main()