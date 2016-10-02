import cv2

img = cv2.imread('fieldnotes_samples_sorted.png')

img = (img >> 4) << 4

cv2.imwrite('fieldnotes_samples_sorted_4bit.png', img)


