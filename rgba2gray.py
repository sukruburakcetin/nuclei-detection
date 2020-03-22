import numpy as np
from PIL import Image
import glob
from natsort import natsorted
import cv2


collated_images = natsorted(glob.glob('C:\\Users\\Burak\\Desktop\\NewNucleiDetection\\masks\\*.png'))
# collated_images = natsorted(glob.glob('C:\\Users\\Burak\\Desktop\\1resize333\\200_mask.png'))

rgbaInputs= list(zip(collated_images))

images = [[np.asarray(Image.open(y)) for y in x] for x in rgbaInputs]


for x in range(len(images)):
    cv2.imwrite('C:\\Users\\Burak\\Desktop\\NewNucleiDetection\\masks2\\' + str(x) + ".png", cv2.cvtColor(images[x][0], cv2.COLOR_BGR2GRAY))






