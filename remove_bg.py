import cv2
import numpy as np

def remove_bg(path):
    # Parameters
    blur = 21  # "smoothness" of the fg-bg dividing line
    canny_low = 15  # min intesity along which edges are drawn, too low -> too many edges
    canny_high = 150  # max intesity along which edges are drawn, contrast above this value -> edge
    min_area = 0.0005  # min area that a contour in fg can occupy [0, 1]
    max_area = 0.95  # max area that a contour in fg can occupy [0, 1]
    dilate_iter = 10  # number of iteration of dilation
    erode_iter = 10  # number of iterations of erosion

    img = cv2.imread(path)

    # Convert image to grayscale
    image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Canny Edge Dection
    edges = cv2.Canny(image_gray, canny_low, canny_high)

    # optional
    edges = cv2.dilate(edges, None)
    edges = cv2.erode(edges, None)

    # Get the area of the image as a comparison
    image_area = img.shape[0] * img.shape[1]

    # get the contours and their areas
    contour_info = [(c, cv2.contourArea(c)) for c in cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]]

    # calculate max and min areas in terms of pixels
    max_area = max_area * image_area
    min_area = min_area * image_area

    # Set up mask with a matrix of 0's
    mask = np.ones(img.shape, dtype=np.uint8) * 255

    # Go through and find relevant contours and apply to mask
    for contour in contour_info:
        # Instead of worrying about all the smaller contours,
        # if the area is smaller than the min, the loop will break
        if min_area < contour[1] < max_area:
            # Add contour to mask
            mask = cv2.fillConvexPoly(mask, contour[0], (0, 0, 0))

    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # negate mask
    mask = 255 - mask

    # apply morphology to remove isolated extraneous noise
    # use border constant of black since foreground touches the edges
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # anti-alias the mask -- blur then stretch
    # blur alpha channel
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=2, sigmaY=2, borderType=cv2.BORDER_DEFAULT)

    # linear stretch so that 127.5 goes to 0, but 255 stays 255
    mask = (2 * (mask.astype(np.float32)) - 255.0).clip(0, 255).astype(np.uint8)

    # put mask into alpha channel
    result = img.copy()
    result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
    result[:, :, 3] = mask
    
    cv2.imwrite('images_no_bg/compost1_no_bg.png', result)

    cv2.destroyAllWindows()
    return result

def main():
    remove_bg('images/compost1.jpg')


if __name__ == '__main__':
    main()


