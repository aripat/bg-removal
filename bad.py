import numpy as np
import cv2
import os
from PIL import Image

BASE_DIR = "C:/Users/aripa/PycharmProjects/bg_removal"
os.chdir(BASE_DIR)


def remove_bg(path):
    # Parameters
    blur = 21  # "smoothness" of the fg-bg dividing line
    canny_low = 15  # min intesity along which edges are drawn, too low -> too many edges
    canny_high = 150  # max intesity along which edges are drawn, contrast above this value -> edge
    min_area = 0.0005  # min area that a contour in fg can occupy [0, 1]
    max_area = 0.95  # max area that a contour in fg can occupy [0, 1]
    dilate_iter = 10  # number of iteration of dilation
    erode_iter = 10  # number of iterations of erosion
    mask_color = (1, 1, 1)  # the color of the background once it is removed

    img = cv2.imread(path)

    # cv2.imshow("Image", img)
    # cv2.waitKey(0)

    # Convert image to grayscale
    image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # cv2.imshow("Foreground", image_gray)
    # cv2.waitKey(0)

    # Apply Canny Edge Dection
    edges = cv2.Canny(image_gray, canny_low, canny_high)

    # optional
    edges = cv2.dilate(edges, None)
    edges = cv2.erode(edges, None)

    # cv2.imshow("Foreground", edges)
    # cv2.waitKey(0)

    # Get the area of the image as a comparison
    image_area = img.shape[0] * img.shape[1]

    # get the contours and their areas
    contour_info = [(c, cv2.contourArea(c)) for c in cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]]

    # calculate max and min areas in terms of pixels
    max_area = max_area * image_area
    min_area = min_area * image_area

    # Set up mask with a matrix of 0's
    mask = np.zeros(img.shape, dtype=np.uint8)

    # Go through and find relevant contours and apply to mask
    for contour in contour_info:
        # Instead of worrying about all the smaller contours,
        # if the area is smaller than the min, the loop will break
        if min_area < contour[1] < max_area:
            # Add contour to mask
            mask = cv2.fillConvexPoly(mask, contour[0], (255, 255, 255))

    # cv2.imshow("Foreground", mask)
    # cv2.waitKey(0)

    # use dilate, erode, and blur to smooth out the mask
    mask = cv2.dilate(mask, None, iterations=dilate_iter)
    mask = cv2.erode(mask, None, iterations=erode_iter)
    mask = cv2.GaussianBlur(mask, (blur, blur), 0)

    # cv2.imshow("Foreground", mask)
    # cv2.waitKey(0)

    # Ensures data types match up
    mask_stack = mask.astype('float32') / 255.0
    img = img.astype('float32') / 255.0

    # Blend the image and the mask
    masked = (mask_stack * img) + ((1 - mask_stack) * mask_color)
    masked = (masked * 255).astype('uint8')

    cv2.imshow("Foreground", masked)
    cv2.waitKey(0)

    # save resulting masked image without bg
    png = to_png(masked)
    cv2.imwrite('images_no_bg\compost1_no_bg.png', png)

    path = r'C:/Users/aripa/PycharmProjects/bg_removal/backgrounds/bg1.jpg'
    bg = cv2.imread(path)
    bg = cv2.resize(bg, (400, 300))
    overlay = overlay_png(png, bg)
    cv2.imwrite('images_new_bg/compost1_new_bg.png', overlay)


def overlay_png(foreground, background, pos=(0,0)):
    # function:: alpha blending: overlay a given png image (RGBA) over given background
    # foreground:: foreground image
    # background:: background image
    # position:: position where the png should be attached
    # output:: image result

    h,w,_ = np.shape(foreground)
    rows,cols,_ = np.shape(background)
    y,x = pos[0],pos[1]

    # apply alpha blending pixel by pixel
    for i in range(h):
        for j in range(w):
            if x+i >= rows or y+j >= cols:
                continue
            alpha = float(foreground[i][j][3] / 255.0)  # take pixel's alpha and make it in rage (0, 1)
            background[x+i][y+j] = alpha*(foreground[i][j][:3])+(1-alpha)*(background[x+i][y+j])     # alpha blending

    return background


def to_png(img):
    # convert to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # threshold input image as mask
    # cv2.THRESH_BINARY:: all pixel's values smaller than 250 go to 0,
    #                     all pixel's values bigger than 250 go to 255
    mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)[1]

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

    return result


def main():
    path = r'C:/Users/aripa/PycharmProjects/bg_removal/images/compost1.jpg'
    remove_bg(path)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
