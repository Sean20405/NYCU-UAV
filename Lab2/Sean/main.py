import random
import cv2
import numpy as np
import matplotlib.pyplot as plt


def part1a():
    img = cv2.imread('histogram.jpg').astype('float64')
    h, w = img.shape[:2]
    b, g, r = cv2.split(img)
    b_hist = np.zeros(256); g_hist = np.zeros(256); r_hist = np.zeros(256)
    b_acc_perct = np.zeros(256); g_acc_perct = np.zeros(256); r_acc_perct = np.zeros(256)
    b_out = np.zeros(256); g_out = np.zeros(256); r_out = np.zeros(256)

    for i in range(h):
        for j in range(w):
            b_hist[int(b[i, j])] += 1
            g_hist[int(g[i, j])] += 1
            r_hist[int(r[i, j])] += 1
    
    for i in range(256):
        b_acc_perct[i] = b_acc_perct[i - 1] + b_hist[i]
        g_acc_perct[i] = g_acc_perct[i - 1] + g_hist[i]
        r_acc_perct[i] = r_acc_perct[i - 1] + r_hist[i]
        b_out[i] = (b_acc_perct[i] / (h * w)) * 255
        g_out[i] = (g_acc_perct[i] / (h * w)) * 255
        r_out[i] = (r_acc_perct[i] / (h * w)) * 255
    
    for i in range(h):
        for j in range(w):
            b[i, j] = b_out[int(b[i, j])]
            g[i, j] = g_out[int(g[i, j])]
            r[i, j] = r_out[int(r[i, j])]
    
    img = cv2.merge((b, g, r))
    cv2.imwrite('part1a.jpg', img.astype(np.uint8))
    
def part1b():
    img = cv2.imread('histogram.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype('float64')
    h, w = img.shape[:2]
    hue, s, v = cv2.split(img)
    v_hist = np.zeros(256)
    v_acc_perct = np.zeros(256)
    v_out = np.zeros(256)

    for i in range(h):
        for j in range(w):
            v_hist[int(v[i, j])] += 1

    for i in range(256):
        v_acc_perct[i] = v_acc_perct[i - 1] + v_hist[i]
        v_out[i] = (v_acc_perct[i] / (h * w)) * 255
    
    for i in range(h):
        for j in range(w):
            v[i, j] = v_out[int(v[i, j])]
    
    img = cv2.merge((hue, s, v)).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    cv2.imwrite('part1b.jpg', img)

def part2():
    img = cv2.imread('input.jpg', cv2.IMREAD_GRAYSCALE)
    h, w = img.shape[:2]

    # Calculate histogram
    hist = np.zeros(256)
    for i in range(h):
        for j in range(w):
            hist[img[i, j]] += 1
    # plt.plot(hist)
    # plt.show()

    # Find the threshold
    best_sigma = 0
    best_thr = 0
    for thr in range(1, 255):
        n1 = np.sum(hist[:thr]); n2 = np.sum(hist[thr:])
        mu1 = np.sum([i * hist[i] for i in range(thr)]) / n1
        mu2 = np.sum([i * hist[i] for i in range(thr, 256)]) / n2
        
        sigma_between = n1 * n2 * (mu1 - mu2) ** 2
        if sigma_between > best_sigma:
            best_sigma = sigma_between
            best_thr = thr

    print('Best threshold:', best_thr)
    # Binarize the image
    for i in range(h):
        for j in range(w):
            img[i, j] = 255 if img[i, j] >= best_thr else 0
    
    img = img.astype(np.uint8)
    cv2.imwrite('part2.jpg', cv2.merge((img, img, img)))
    return img

def part3():
    img = part2()
    h, w = img.shape[:2]
    labeled_img = np.zeros((h, w), dtype=np.int32)
    label_cnt = 1
    label_corr = [-1]

    def find(i):
        if label_corr[i] < 0:
            return i
        else:
            label_corr[i] = find(label_corr[i])
            return label_corr[i]

    def union(i, j):
        root_i = find(i)
        root_j = find(j)

        if root_i == root_j:
            label_corr[root_j] += label_corr[root_i]
            label_corr[root_i] = label_corr[root_j]
        elif root_i > root_j:
            label_corr[root_j] += label_corr[root_i]
            label_corr[root_i] = root_j
        else:
            label_corr[root_i] += label_corr[root_j]
            label_corr[root_j] = root_i

    for i in range(h):
        for j in range(w):
            if img[i, j] == 255:
                # Edge case
                if i == 0 and j == 0:
                    labeled_img[i, j] = label_cnt
                    label_corr.append(-1)
                    label_cnt += 1
                elif i == 0:
                    if labeled_img[i, j-1] == 0:
                        labeled_img[i, j] = label_cnt
                        label_corr.append(-1)
                        label_cnt += 1
                    else:
                        labeled_img[i, j] = labeled_img[i, j-1]
                elif j == 0:
                    if labeled_img[i-1, j] == 0:
                        labeled_img[i, j] = label_cnt
                        label_corr.append(-1)
                        label_cnt += 1
                    else:
                        labeled_img[i, j] = labeled_img[i-1, j]
                # General case
                else:
                    # Label of left and top are both 0
                    if labeled_img[i-1, j] == 0 and labeled_img[i, j-1] == 0:
                        label_corr.append(-1)
                        labeled_img[i, j] = label_cnt
                        label_cnt += 1

                    # One of top and left has label
                    elif labeled_img[i-1, j] == 0:
                        labeled_img[i, j] = labeled_img[i, j-1]
                    elif labeled_img[i, j-1] == 0:
                        labeled_img[i, j] = labeled_img[i-1, j]

                    # Label of left and top are not the same
                    elif labeled_img[i-1, j] != labeled_img[i, j-1]:
                        union(labeled_img[i-1, j], labeled_img[i, j-1])
                        labeled_img[i, j] = find(labeled_img[i-1, j])
                    else:
                        labeled_img[i, j] = labeled_img[i-1, j]
    
    # Pass 2: Perform label translation
    for i in range(h):
        for j in range(w):
            labeled_img[i, j] = find(labeled_img[i, j])

    # Color
    label_color = [[random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)] for _ in range(label_cnt)]
    color_img = np.zeros((h, w, 3))
    for i in range(h):
        for j in range(w):
            color_img[i, j] = label_color[labeled_img[i, j]] if labeled_img[i, j] != 0 else [0, 0, 0]

    cv2.imwrite('part3.jpg', color_img.astype(np.uint8))

if __name__ == '__main__':
    # part1a()
    # part1b()
    # part2()
    part3()