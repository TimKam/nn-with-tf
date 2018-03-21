import numpy as np
import matplotlib.image as img
from PIL import Image
import scipy.misc
import scipy.ndimage
import math
import cv2
import numpy as np
from matplotlib import pyplot as plt


def filter_image(image):
    """
    Applies Scharr low pass filter to an immage
    :param image: Image, to which the filter should be applied
    :return:
    """
    kernel = np.ndarray(shape=(3, 3), buffer=np.array([[3, 0, -3],[10, 0, -10],[3, 0, -3]]), dtype=np.uint8)
    filtered_image = cv2.filter2D(image,-1,kernel)
    return filtered_image

"""plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(dst),plt.title('Filtered')
plt.xticks([]), plt.yticks([])
plt.show()"""

# filter_image('../images/view0.png')


def compute_gradient(image):
    """
    Returns the first-order derivative of an image
    :param image: image, for which the gradient should be determined
    :return: first-order derivative of an image
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # load and convert to greyscale
    # derivative = cv2.cvtColor(cv2.imread('../images/view0.png'), cv2.COLOR_BGR2GRAY)  # use as image template
    derivative = np.ndarray(shape=(image.shape[0], image.shape[1], 2), dtype=np.float32)
    row_index = 0
    for row in image:
        column_index = 0
        for pixel in row:
            try:
                derivative.itemset(
                    row_index,
                    column_index,
                    0,
                    image[row_index - 1][column_index] - image[row_index + 1][column_index]
                )
                derivative.itemset(
                    row_index,
                    column_index,
                    1,
                    image[row_index][column_index - 1] - image[row_index + 1][column_index]
                )
                """abs(image[row_index-1][column_index] - image[row_index+1][column_index]) +
                abs(image[row_index][column_index-1] - image[row_index+1][column_index])"""
            except IndexError: # set derivative to 0 on image border pixels
                derivative.itemset(row_index, column_index, 0, 0)
                derivative.itemset(row_index, column_index, 1, 0)
            column_index +=1
        row_index += 1

    return filter_image(derivative)

# image = np.asarray(Image.open('../images/view0.png'))
# derivative = compute_gradient(cv2.imread('../images/view0.png'))
# print(derivative)
# cv2.imwrite('../images/greyscale.png', cv2.cvtColor(cv2.imread('../images/view0.png'), cv2.COLOR_BGR2GRAY))
# cv2.imwrite('../images/view0_out.png', derivative)
# cv2.imwrite('../images/view0_filtered.png', filter_image(image))


def estimate_derivative_matrix(image, position, region_size_x=21, region_size_y=21):
    """
    :param image: image, for which the derivative matrix should be estimated
    :param position: position of the center pixel of the region
    :param region_size_x: x size of the region in pixel
    **Note:** If the region size is even, it will be extended by one pixel (to have an actual center pixel)
    :param region_size_y: y size of the region in pixel
    **Note:** If the region size is even, it will be extended by one pixel (to have an actual center pixel)
    :return: derivative for region
    """
    x_lower_boundary = position[0] - math.floor(region_size_x / 2)
    x_upper_boundary = position[0] + math.floor(region_size_x / 2)
    y_lower_boundary = position[1] - math.floor(region_size_y / 2)
    y_upper_boundary = position[1] + math.floor(region_size_y / 2)
    gradient = compute_gradient(image)
    return gradient[x_lower_boundary:x_upper_boundary, y_lower_boundary:y_upper_boundary]


# derivative_matrix = estimate_derivative_matrix(cv2.imread('../images/view0.png'), [100, 100], 20)
# print(derivative_matrix)


def estimate_frame_difference(image_0, image_1, position, region_size_x=21, region_size_y=21):
    """
    :param image_0: original image
    :param image_1: image at next point in time
    :param position: position of the center pixel of the region
    :param region_size_x: x size of the region in pixel
    **Note:** If the region size is even, it will be extended by one pixel (to have an actual center pixel)
    :param region_size_y: y size of the region in pixel
    **Note:** If the region size is even, it will be extended by one pixel (to have an actual center pixel)
    :return:
    """
    image_0 = cv2.cvtColor(image_0, cv2.COLOR_BGR2GRAY)
    image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
    x_lower_boundary = position[0] - math.floor(region_size_x / 2)
    x_upper_boundary = position[0] + math.floor(region_size_x / 2)
    y_lower_boundary = position[1] - math.floor(region_size_y / 2)
    y_upper_boundary = position[1] + math.floor(region_size_y / 2)
    region_image_0 = image_0[x_lower_boundary:x_upper_boundary, y_lower_boundary:y_upper_boundary]
    region_image_1 = image_1[x_lower_boundary:x_upper_boundary, y_lower_boundary:y_upper_boundary]
    frame_difference = region_image_0 - region_image_1
    return frame_difference.astype(np.float32)


# Least square
def determine_displacement(derivative_matrix, frame_difference):
    """
    Determines the displacement vector using least squares regression
    :param derivative_matrix:
    :param frame_difference:
    :return:
    """
    derivative_matrix_2d = derivative_matrix.reshape(derivative_matrix.shape[0] * derivative_matrix.shape[1], 2)
    frame_difference_2d = frame_difference.reshape(frame_difference.shape[0] * frame_difference.shape[1])
    vector = np.linalg.lstsq(derivative_matrix_2d, frame_difference_2d)
    return vector

# Determine new position
def interpolate():

    new_position = ''
    return new_position



image_0 = cv2.imread('../images/view0.png')
image_1 = cv2.imread('../images/view1.png')
difference = estimate_frame_difference(image_0, image_1, [336, 320])
derivative_matrix = estimate_derivative_matrix(image_0, [336, 320])
vector = determine_displacement(derivative_matrix, difference)
print(vector[0])

image_0 = cv2.cvtColor(cv2.imread('../images/view0.png'), cv2.COLOR_BGR2GRAY)
image_1 = cv2.cvtColor(cv2.imread('../images/view1.png'), cv2.COLOR_BGR2GRAY)

pointsToTrack = np.ndarray(shape=(1, 2), buffer=np.array([[336, 320]]), dtype=np.float32)
p1, st, err = cv2.calcOpticalFlowPyrLK(image_0, image_1, pointsToTrack, pointsToTrack)
print(p1)


"""print('Difference: ' + str(difference))
print('Shape: ' + str(difference.shape))"""

### Interpolation function



### Final tracker

