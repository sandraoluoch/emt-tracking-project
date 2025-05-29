from bioio import BioImage
import bioio_ome_tiff
from skimage.transform import rescale, resize
import tifffile
from tifffile import imsave
from bioio.writers import OmeTiffWriter
import numpy as np
from tqdm import tqdm

input_path = "/allen/aics/assay-dev/users/Sandi/emt-tracking/emt-data-013025/seg/3500006851/B3_P0/movie/8bit_mask_0-200.tiff"
output_path = "/allen/aics/assay-dev/users/Sandi/emt-tracking/emt-data-013025/seg/3500006851/B3_P0/"
downsampling_factor=2

# read in image to get shape 
img = BioImage(input_path) 
# img.data
print("full image shape:", img.shape) # "TCZYX"

# Calculate new dimensions
new_shape = (
        int(img.shape[0]),
        int(img.shape[1]),
        int(img.shape[2]),
        int(img.shape[3]) / downsampling_factor,
        int(img.shape[4]) / downsampling_factor,
    )

list_of_imgs =[]

# for t in tqdm(range(0,2)):
for t in tqdm(range(img.shape[0])):  # Iterate over time
    # Downsample using resize
    IMG_at_t = img.get_image_data('ZYX', T=t, C=0)
    downsampled_image = resize(IMG_at_t, new_shape, order=0, anti_aliasing=False)
    list_of_imgs.append(downsampled_image)
    # OmeTiffWriter.save(downsampled_image, output_path + "seg_downsampled_2x_" + "{}".format(t) + ".tiff", dim_order="TCZYX")


downsampled_image = np.concatenate(list_of_imgs, axis=0)


print("downsampled img shape:", downsampled_image.shape)


OmeTiffWriter.save(downsampled_image, output_path + "seg_downsampled_2x_0-200.tiff", dim_order="TCZYX")
print("Downsampled image saved!")



