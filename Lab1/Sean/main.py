import cv2
import numpy as np
from sklearn.preprocessing import binarize

def part1_1():
    img = cv2.imread('test.jpg').astype('float64')
    h, w = img.shape[:2]
    for i in range(h):
        for j in range(w):
            b = img[i, j, 0]
            g = img[i, j, 1]
            r = img[i, j, 2]
            if not (b > 100 and b * 0.6 > g and b * 0.6 > r):
                inten = (b + g + r) / 3
                img[i, j] = [inten, inten, inten]
    cv2.imwrite('part1_1.jpg', img.astype('uint8'))

def part1_2():
    img = cv2.imread('test.jpg').astype(np.int32)
    h, w = img.shape[:2]
    CONTRAST = 100.0
    BRIGHTNESS = 40.0
    for i in range(h):
        for j in range(w):
            b, g, r = img[i, j, 0], img[i, j, 1], img[i, j, 2]
            # Case 1: detect yellow
            # if ((r + g) > 250 and abs(r - g) < 50 and (r + g) > 3 * b) or (b > 100 and b * 0.6 > g and b * 0.6 > r):
            
            # Case 2: given condition
            if ((b + g) * 0.3 > r) or (b > 100 and b * 0.6 > g and b * 0.6 > r):
                # img[i, j] = [255, 255, 255]
                img[i, j] = (img[i, j] - 127) * (CONTRAST / 127 + 1) + 127 + BRIGHTNESS

    img = np.clip(img, 0, 255)
    cv2.imwrite('part1_2.jpg', img.astype(np.uint8))

def part2(img_path, scale):
    def bilinear_interpolation(img, i, j):
        h, w = img.shape[:2]
        x1 = int(np.floor(i / scale)); y1 = int(np.floor(j / scale))
        x2 = x1 + 1; y2 = y1 + 1

        # edge case
        if x2 >= h or y2 >= w:
            return img[x1, y1]
        
        Q11 = img[x1, y1]; Q12 = img[x1, y2]; Q21 = img[x2, y1]; Q22 = img[x2, y2]
        i /= scale; j /= scale
        R1 = (x2 - i) / (x2 - x1) * Q11 + (i - x1) / (x2 - x1) * Q21
        R2 = (x2 - i) / (x2 - x1) * Q12 + (i - x1) / (x2 - x1) * Q22
        P = (y2 - j) / (y2 - y1) * R1 + (j - y1) / (y2 - y1) * R2
        return P
    
    img = cv2.imread(img_path).astype('float64')
    h, w = img.shape[:2]
    new_h = h * scale
    new_w = w * scale
    new_img = np.zeros((new_h, new_w, 3))
    for i in range(new_h):
        for j in range(new_w):
            new_img[i, j] = bilinear_interpolation(img, i, j)
    cv2.imwrite('part2.jpg', new_img.astype('uint8'))

def part3():
    img = cv2.imread('ive.jpg', cv2.IMREAD_GRAYSCALE).astype(np.float64)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    gx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    gy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    Gx = cv2.filter2D(img, -1, gx)
    Gy = cv2.filter2D(img, -1, gy)
    G = np.sqrt(Gx ** 2 + Gy ** 2)
    # G = binarize(G, threshold=100) * 255
    cv2.imwrite('part3.jpg', G.astype(np.uint8))
    cv2.imwrite('part3_gx.jpg', Gx.astype(np.uint8))
    cv2.imwrite('part3_gy.jpg', Gy.astype(np.uint8))
    cv2.imwrite('part3_img.jpg', img.astype(np.uint8))


if __name__ == '__main__':
    # part1_1()
    # part1_2()
    # part2('ive.jpg', 3)
    part3()