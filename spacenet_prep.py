"""
The SpaceNet imagery is distributed as 16-bit imagery, 
and the road network is distributed as a line-string vector GeoJSON format.
This script converts the SpaceNet imagery to 8-bit, creates the road masks and distributes all files to final folders.
"""

import raster_utils
import json
import os
import glob
from matplotlib import pyplot
import numpy as np

def main():
    with open('config.json') as json_file:

        # open the config.json file that contains all the necessary information
        config = json.load(json_file)

        # iterate through every region and convert images to 8-bit
        for region, dirs in config['pre-processing']['SpaceNet']['regions'].items():
            print('*********************************')
            print('Preparing region ', region, '...')
            keys = list(dirs)
            # creates a mosaic to compute global histogram of bands
            mosaic, _ = raster_utils.create_mosaic(dirs[keys[0]], dirs[keys[1]])
            mask = np.where(mosaic == 0, mosaic, np.nan)
            masked_mosaic = np.ma.masked_array(mosaic, mask)

            mean = masked_mosaic.mean()
            std = masked_mosaic.std()


            #playground
            red_mean = np.mean(mosaic[0])
            red_std = np.std(mosaic[0])
            '''
            green_mean = np.mean(mosaic[1])
            green_std = np.std(mosaic[1])

            blue_mean = np.mean(mosaic[2])
            blue_std = np.std(mosaic[2])



           
            hist = raster_utils.compute_hist_for_single_image(mosaic, bins='auto')
            print(hist[0])
            n, bins, patches = plt.hist(mosaic, raster_utils.equal_height_bins_hist(mosaic, 10))
            print('n= ', n)
            print('bins= ', bins)
            print('patches= ', patches)
            break


            # Conversion of training and testing images
            search_criterion = "*.tif"
            q1 = os.path.join(dirs[keys[0]], search_criterion)
            train_src_images = glob.glob(q1)
            q2 = os.path.join(dirs[keys[1]], search_criterion)
            test_src_images = glob.glob(q2)
            train_src_images = []
            testing_src_images = []

            for train_image in train_src_images:
                pass

            for test_image in test_src_images:
                pass


            print('Finished with region ', region)
            '''
            break

    print('SpaceNet dataset prep is done!')

if __name__ == "__main__":
# execute only if run as a script
    main()