import numpy as np
import matplotlib.pyplot as plt
from skimage import io, exposure, filters, segmentation, measure, morphology, \
                    feature
from scipy import ndimage
from thresholding import apply_hysteresis_threshold
import os

def crop_afm_bar(im):
    """
    Crop image to remove blank borders and colorbar. Are the dimensions
    always the same? To be checked
    """
    return im[22:980, 5:965]

# We use both the phase and height images
name_phase = 'E845_phase0.jpg'
name_height = 'E845_height0.jpg'
im1 = crop_afm_bar(io.imread(name_phase, as_grey=True))
imh = crop_afm_bar(io.imread(name_height, as_grey=True))
# Keep original images before filtering and masking low regions
im1_orig = np.copy(im1)
imh_orig = np.copy(imh)

# Median filtering to smooth out noise
im1 = ndimage.median_filter(im1, 5)
imh = ndimage.median_filter(imh, 5)

# Set to zero regions of low elevation
mask = np.logical_not(apply_hysteresis_threshold(imh, 0.1, 0.15))
im1[mask] = 0
imh[mask] = 0

# Detect local maxima so that (ideally) there is one in each grain
# We use a linear combination of both modalities: we're looking mostly
# for height maxima but some edge enhancement due to phase also helps
peaks = feature.peak_local_max(imh + 0.1 * im1, min_distance=5, indices=False)
label_peaks = measure.label(peaks)

# Watershed segmentation
labels = morphology.watershed(-im1, label_peaks, mask=(im1 > 0))
labels = segmentation.clear_border(labels)
new_order = np.random.permutation(np.arange(0, labels.max() + 1))
new_labels = new_order[labels]

# plot results
fig, ax = plt.subplots(ncols=2, nrows=2, sharex=True, sharey=True)
ax[0, 0].imshow(im1_orig, cmap='gray')
ax[0, 1].imshow(segmentation.mark_boundaries(im1_orig, labels))
ax[1, 0].imshow(imh_orig, cmap='gray')
ax[1, 0].contour(peaks, colors='yellow')
ax[1, 1].imshow(new_labels)

plt.show()

save_name = os.path.splitext(name_phase)[0] + '_labels.npy'
np.save(save_name, labels)
