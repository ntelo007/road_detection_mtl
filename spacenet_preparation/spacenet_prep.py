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
            t1 = perf_counter()
            print('Preparing region ', region, '...')
            keys = list(dirs)

            # if you want to measure time for every function
            # use start = time.clock() and end= .. between every function

            # creates a raster mosaic of all the satellite images withing the test and train directories
            print('Preparing mosaic...')
            mosaic, _ = raster_utils.create_mosaic(dirs[keys[0]], None)    #the second dir might be useless -> no gt exists
            
            # computes global histogram boundaries that contain 99% of the data
            print('Calculating boundaries using histogram...')
            boundaries = raster_utils.compute_min_max_from_cdf(mosaic)
            
            # convert images to 8 bit and save them all in the train folder of SpaceNet
            print('Converting images to 8 bit...')
            if not os.path.isdir(dirs[keys[3]]):
                os.makedirs(dirs[keys[3]])
            raster_utils.convert_16_to_8_bits(dirs[keys[0]], boundaries, dirs[keys[3]], region)
            
            # creation of road masks
            print("Creating road masks...")
            raster_utils.create_road_masks(dirs[keys[0]], dirs[keys[2]], dirs[keys[4]], region)

            print("Creating crops...")
            raster_utils.create_crops(output_crops_path=dirs[keys[5]],
                                      uncropped_masks=dirs[keys[4]],
                                      uncropped_imgs=dirs[keys[3]],
                                      size=config['pre-processing']['SpaceNet']['crop_size'],
                                      overlap=config['pre-processing']['SpaceNet']['crop_overlap'],
                                      image_suffix=config['pre-processing']['SpaceNet']['image_suffix'],
                                      gt_suffix=config['pre-processing']['SpaceNet']['gt_suffix']
                                      )
            # distribute images from the train, test and val dirs with random selections
            print("Distributing images to train/val/test folders...")
            raster_utils.distribute_images(config, dirs)
            t2 = perf_counter()
            print('Finished with region {} in {}s'.format(region, t2-t1))

    # Stop the stopwatch / counter
    t1_stop = perf_counter()

    print('\n\nSpaceNet dataset prep is done! Time elapsed: {0}s'.format(t1_stop-t1_start))


if __name__ == "__main__":
# execute only if run as a script
    main()