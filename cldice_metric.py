# centerline-in-mask-dice-coefficient or clDice
# connectivity-preserving metric to evaluate tubular and linear structure segmentation based on intersecting skeletons with masks.

from skimage import io, morphology, filters, util
import numpy as np
import os
from skimage.viewer import ImageViewer
import matplotlib.pyplot as plt


def binarize(img):
    if len(np.unique(img)) > 1:
        img = img > filters.threshold_otsu(img)
    return img

def cl_score(v, s):
    return np.sum(v*s) / np.sum(s)

# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()

def clDice(v_p, v_l):
    if len(np.unique(v_l)) > 1:
        if len(np.unique(v_p)) > 1:

            v_p = v_p > filters.threshold_otsu(v_p)
            v_l = v_l > filters.threshold_otsu(v_l)

            s_l = morphology.skeletonize(v_l)
            s_p = morphology.skeletonize(v_p)

            # visualize(
            #     Ground_Truth = v_l,
            #     gt_skeleton = s_l,
            #     Prediction = v_p,
            #     pr_skeletong = s_p
            # )

            tprec = cl_score(v_p, s_l)
            tsens = cl_score(v_l, s_p)
            clDice = 2*tprec*tsens/(tprec+tsens+0.001)
            return clDice, 0
        else:
            return 0, 0
    else:
        return 1, 1

# # example usage
# base_dir = 'C:\\Users\\kaniourasp\\Downloads\\Thesis\\APLS\\New folder\\road_visualizer-master\\data\\nteloSample'
# gt_file = os.path.join(base_dir, '2_gt.png')
# pred_file = os.path.join(base_dir, '2_pred.png')

# gt = io.imread(gt_file)[:, :, 0]
# pred = io.imread(pred_file)[:, :, 0]

# # change foreground with background if needed
# binary_gt = util.invert(binarize(gt))
# binary_pred = util.invert(binarize(pred))

# io.imshow(binary_gt)
# io.show()
# io.imshow(binary_pred)
# io.show()

# measure = clDice(binary_pred, binary_gt)

# print("measure: ", measure)


