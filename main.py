import glob
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

image_set = "Coast"  # this should be Forest, Coast, or bedroom

image_names = [path.rsplit('\\', 1)[-1] for path in glob.glob('./ProjData/Train/' + image_set + '/*.jpg')]

images = [cv.imread(file, cv.IMREAD_GRAYSCALE) for file in glob.glob('./ProjData/Train/' + image_set + '/*.jpg')]

start = 20
end = 24

for i in range(len(images)):
    x = images[i].shape[0]
    y = images[i].shape[1]
    current_brightness = np.sum(images[i]) / (255 * x * y)

    if (current_brightness < 0.4 or current_brightness > 0.6):

        while(current_brightness < 0.4 or current_brightness > 0.6):
            brightness_adjustment = round(((255 * x * y * 0.5) - np.sum(images[i])) / (x * y))
            adjustment_array = np.ones((x, y), dtype=int) * brightness_adjustment
            images[i] = np.clip(np.add(images[i], adjustment_array), 0, 255)
            current_brightness = np.sum(images[i]) / (255 * x * y)


images_at_200 = images.copy()
images_at_50 = images.copy()

for x in range(len(images)):
    images_at_200[x] = cv.resize(images[x], (200,200), interpolation=cv.INTER_LINEAR_EXACT).astype(np.uint8)
    images_at_50[x] = cv.resize(images[x], (50,50), interpolation=cv.INTER_LINEAR_EXACT).astype(np.uint8)
    cv.imwrite("./AnalysisData/Train/" + image_set + "/200/" + image_names[x], images_at_200[x])
    cv.imwrite("./AnalysisData/Train/" + image_set + "/50/" + image_names[x], images_at_50[x])

plt.subplot(311), plt.imshow(images[3], 'gray'),plt.title("original")
plt.subplot(312), plt.imshow(images_at_200[3], 'gray'),plt.title("200")
plt.subplot(313), plt.imshow(images_at_50[3], 'gray'),plt.title("50")
plt.show()
