import cv2
import numpy as np

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
    part3()