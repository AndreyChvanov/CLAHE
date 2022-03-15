import numpy as np
import matplotlib.pyplot as plt
import ahe
import cv2

img = ahe.load_image('images/AHE.png')
process_img = ahe.AHE(img)
plt.imshow(process_img, cmap='gray')
plt.show()
cv2.imwrite("result.png", process_img)

