import cv2
import numpy as np
import math

# Global variables to store the starting and ending points of the line
drawing = False  # True if the mouse is pressed
ix, iy = -1, -1
ex, ey = -1, -1

# Mouse callback function
def draw_line(event, x, y, flags, param):
    global ix, iy, drawing, image, ex, ey

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        print(f"Starting point: ({ix}, {iy})")

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            # Create a copy of the image to draw the line dynamically
            img_copy = image.copy()
            cv2.line(img_copy, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.imshow('image', img_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        ex, ey = x, y
        length_pixels = math.sqrt((ex-ix)**2 + (ey-iy)**2)
        cv2.line(image, (ix, iy), (ex, ey), (0, 255, 0), 2)
        cv2.putText(image, f"{length_pixels:.2f} pixels", (ex, ey-5), cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5, color=(0, 255, 0), thickness=2)
        print(f"Ending point: ({ex}, {ey})")
        print(f"The length is {length_pixels:.2f} pixels")
        cv2.imshow('image', image)

# Load an image
image = cv2.imread('picture_0.jpg')

# Create a window and set the mouse callback function
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_line)

# Display the image and wait for a key press
cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()