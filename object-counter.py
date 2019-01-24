# upload required dependencies
import cv2
import numpy as np
import math
import copy

# upload the video
cap = cv2.VideoCapture('Dataset/DatasetC.avi')

#  we are going to apply gaussian filer to smooth out the image to account for changes in intensity
#  which were not caused by motion
gaussianKernel = np.matrix('1 2 1; 2 4 2; 1 2 1')  # going to apply 3z3 as more detail is preserved
gaussianKernel1 = np.matrix('1 4 7 4 1; 4 16 26 16 4; 7 26 41 26 7; 4 16 26 16 4; 1 4 7 4 1')

frame_count = 0  # initialing the count to find consecutive frames
frame_one = []  # initialising the variable to hold the first frame
frame_two = []  # initialising the variable to hold the second frame


#  here we define the function which applies a filer to an image, taken from the Assignment 1
def icv_filter(image, kernel):
    kernel_array = kernel.getA()
    kernel_size = 0
    kernel_dimension = int(math.sqrt(kernel_array.size))
    width = image[0].size
    height = int(image.size / width)

    # calculate the size of the kernel and its factor
    for i in range(0, kernel_dimension):
        for j in range(0, kernel_dimension):
            kernel_size = kernel_size + kernel_array[i][j]

    if kernel_size != 0:
        kernel_factor = 1 / kernel_size
    else:
        kernel_factor = 1
    # create canvas for a new image
    new_image = np.full((height, width, 1), 255, np.uint8)

    # looping over the image
    for i in range(0, height):
        for j in range(0, width):
            intensity = 0
            for ki in range(0, kernel_dimension):
                for kj in range(0, kernel_dimension):
                    image_x = int((j - kernel_dimension / 2 + 0.5 + kj))
                    image_y = int((i - kernel_dimension / 2 + 0.5 + ki))
                    if image_x < 0 or image_y < 0 or image_y >= height or image_x >= width:
                        intensity += 0
                    else:
                        intensity += image[image_y][image_x] * kernel_array[ki][kj]

            new_image[i][j] = min(max(int(kernel_factor * intensity), 0), 255)

    return new_image


#  loads frames, in order to run icv_generate_reference_frame comment out the code block below
while cap.isOpened():
    ret, frame = cap.read()
    #  convert the image to gray-scale since color has no effect on motion detection algorithms
    if frame_count == 95:  # recording the first frame
        frame_one = icv_filter(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), gaussianKernel)
    if frame_count == 130:  # recording the second frame
        frame_two = icv_filter(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), gaussianKernel)
        break
    frame_count += 1
cap.release()


# defining a function to check if the required threshold(change in rgb values) has been reached
def icv_check_threshold(pixel_value):
    if pixel_value < 35:  # change threshold here
        return False  # return False if not satisfied
    else:
        return True  # return True if satisfied


# defining the function for the block matching algorithm
def icv_pixel_frame_differencing(frame_1, frame_2):
    #  first convert frames to numpy arrays to make it easier to work with
    first_frame = np.asarray(frame_1, dtype=np.float32)
    second_frame = np.asarray(frame_2, dtype=np.float32)

    #  then compute frame dimensions
    frame_width = int(first_frame[0].size)
    frame_height = int(first_frame.size/frame_width)

    #  we then create a stock image for differencing output
    frame_difference = np.zeros((frame_height, frame_width), np.uint8)

    for i in range(0, frame_width - 1):
        for j in range(0, frame_height - 1):
            # compute the absolute difference between the current frame and first frame
            frame_difference[j, i] = abs(first_frame[j, i] - second_frame[j, i])
            # check if the threshold = 25 is satisfied, if not set pixel value to 0, else to 255
            # comment out code below to obtain result without threshold / non-binary
            if icv_check_threshold(frame_difference[j, i]):
                frame_difference[j, i] = 255
            else:
                frame_difference[j, i] = 0

    cv2.imwrite("differenceC.jpg", frame_difference)
    cv2.imwrite("frame50.jpg", first_frame)
    cv2.imwrite("frame51.jpg", second_frame)
    return frame_difference


# defining function to generate reference frame
def icv_generate_reference_frame(video):
    # first we initialise all required variables
    reference_frame = 0
    frame_height = 0
    frame_width = 0
    background_frame = 0
    anchor_frame = 0
    first_frame = True
    pixels_used_count = 0  # counts the number of times a pixel has not significantly changed (i.e no movement)
    pixel_value_sum = 0  # adds up pixel values when no significant movement takes place
    while video.isOpened():
        res, video_frame = video.read()
        if not res:
            break
        if first_frame:
            first_frame = False
            #  set first frame as the first reference frame
            reference_frame = icv_filter(cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY), gaussianKernel)
            #  calculate frame dimensions
            frame_width = int(reference_frame[0].size)
            frame_height = int(reference_frame.size / frame_width)
            #  initiate final reference frame and calculation variables
            background_frame = np.zeros((frame_height, frame_width), np.uint8)
            pixels_used_count = np.zeros((frame_height, frame_width), np.uint8)
            pixel_value_sum = np.zeros((frame_height, frame_width), np.uint16)
        else:
            anchor_frame = reference_frame  # here we set the anchor frame to the previous reference frame
            reference_frame = icv_filter(cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY), gaussianKernel).astype(np.uint16)
            for i in range(0, frame_width - 1):
                for j in range(0, frame_height - 1):
                    # compute the absolute difference between the current frame and first frame
                    pixel_difference = abs(reference_frame[j, i] - anchor_frame[j, i])
                    # check if the threshold = 25 is satisfied, if not the pixel can be used to generate background
                    # as no movement takes place
                    if not icv_check_threshold(pixel_difference):
                        pixels_used_count[j, i] += 1
                        pixel_value_sum[j, i] += reference_frame[j, i]

    video.release()

    # block below iterates over all pixels and calculates weighted average
    for i in range(0, frame_width - 1):
        for j in range(0, frame_height - 1):
            background_frame[j, i] = int(pixel_value_sum[j, i] / pixels_used_count[j, i])
            print(background_frame[j, i])

    cv2.imwrite("background.jpg", background_frame)


def icv_count_objects():
    reference = cv2.imread("results/question2C/background.jpg", cv2.IMREAD_GRAYSCALE)
    cap = cv2.VideoCapture('Dataset/DatasetC.avi')

    number_of_objects = []
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        difference = icv_pixel_frame_differencing(reference, icv_filter(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), gaussianKernel))
        frame_width = int(difference[0].size)
        frame_height = int(difference.size / frame_width)
        # the next block will attempt to fill in the blanks
        binary_rotation = [[-1, -1], [0, -1], [1, -1], [1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0]]
        for i in range(0, frame_width - 1):
            for k in range(0, frame_height - 1):
                # first we find lit pixels
                if difference[k, i] == 255:
                    for rotation_i in range(0, 8):  # iterate over surrounding pixels
                        if 0 < i + binary_rotation[rotation_i][1] < frame_width and \
                                0 < k + binary_rotation[rotation_i][0] < frame_height:
                            if difference[
                                k + binary_rotation[rotation_i][0], i + binary_rotation[rotation_i][1]] == 0:
                                lit_neighbour_count = 0
                                for rotation_k in range(0, 8):
                                    if 0 < i + binary_rotation[rotation_i][1] + binary_rotation[rotation_k][
                                        1] < frame_width and \
                                            0 < k + binary_rotation[rotation_i][0] + binary_rotation[rotation_k][
                                        0] < frame_height:
                                        if difference[
                                            k + binary_rotation[rotation_i][0] + binary_rotation[rotation_k][0],
                                            i + binary_rotation[rotation_i][1] + binary_rotation[rotation_k][
                                                1]] == 255:
                                            lit_neighbour_count += 1
                                if lit_neighbour_count > 3: # set threshold for neighbours here
                                    difference[k + binary_rotation[rotation_i][0], i + binary_rotation[rotation_i][
                                        1]] = 255
        # the next block will count the number of objects in the difference frame
        if frame_count == 94:
            cv2.imwrite("filled_up.jpg", difference)
        object_count = 0
        for i in range(0, frame_width - 1):
            for k in range(0, frame_height - 1):
                if difference[k, i] == 255:
                    # if white pixel found count 1 object
                    object_count += 1
                    to_fill = [[k, i]]
                    iterator = 0
                    while len(to_fill) != 0:
                        iterator += 1
                        print(to_fill)
                        difference[to_fill[0][0], to_fill[0][1]] = 100 # fill surrounding pixels gray
                        for rotation_i in range(0, 8):  # iterate over surround pixels
                            if 0 < to_fill[0][1] + binary_rotation[rotation_i][1] < frame_width and \
                                    0 < to_fill[0][0] + binary_rotation[rotation_i][0] < frame_height:
                                # if white pixel found and not yet added to the que, add it to the que
                                if difference[to_fill[0][0] + binary_rotation[rotation_i][0], to_fill[0][1] +
                                                                                                binary_rotation[
                                                                                                    rotation_i][
                                                                                                    1]] == 255 and \
                                        [to_fill[0][0] + binary_rotation[rotation_i][0],
                                         to_fill[0][1] + binary_rotation[rotation_i][1]] not in to_fill:
                                    to_fill.insert(len(to_fill), [to_fill[0][0] + binary_rotation[rotation_i][0],
                                                                to_fill[0][1] + binary_rotation[rotation_i][1]])
                        # remove filled pixel from que
                        to_fill.pop(0)
                    # if no neighboring pixels have been found, discard the pixel
                    # by changing this parameter we can limit the minimum number of pixels, required to count an object
                    if iterator == 1:
                        object_count -= 1
        number_of_objects.append(object_count)
        if frame_count == 94:
            cv2.imwrite("filled_up_g.jpg", difference)
            print(object_count)
        frame_count += 1
        print("got %s" % frame_count)

    cap.release()
    cv2.imwrite("filled_up.jpg", difference)
    print(number_of_objects)
    print(len(number_of_objects))


# comment out when running icv_generate_reference_frame(cap)
# icv_pixel_frame_differencing(frame_one, frame_two)

icv_count_objects()

# uncomment line below to generate reference frame
# icv_generate_reference_frame(cap)
