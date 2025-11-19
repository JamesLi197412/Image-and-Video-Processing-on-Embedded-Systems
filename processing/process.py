from numpy import interp, histogram
from PIL import Image

def histeq(im, nbr_bins=256):
    imhist, bins = histogram(im.flatten(), nbr_bins, normed = True)
    cdf = imhist.cumsum()   # cumulative distribution function
    cdf = 255 * cdf / cdf[-1] # normalize

    # user linear interpolation of cdf to find new pixel values
    im2 = interp(im.flatten(), bins[:-1], cdf)

    return im2.reshape(im.shape), cdf