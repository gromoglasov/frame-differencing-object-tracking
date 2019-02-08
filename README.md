# Frame differencing Object Tracking

My implementation of frame differencing for calculating the number of cars from video footage. When looking through the code
you could notice that I am using my own function for applying kernels, feel free to rewrite it using OpenCV's native function,
this should speed up the run time.

**There are a few things in one file:**
- icv_filter - applies a kernel
- icv_pixel_frame_differencing - frame differencing
- icv_generate_reference_frame - generates a reference frame from a video input
- icv_count_objects - counts the number of objects in each video frame, using the reference frame computed from a video stream

The project is using Python 3.6.

## List of Dependencies:

- OpenCV

- NumPy


**NOT JUPYTER NOTEBOOK**

Run from terminal, PyCharm, or from anything else that makes you happy:)

Everything else you need to know is in the comments.

Star, fork, do your thing.
