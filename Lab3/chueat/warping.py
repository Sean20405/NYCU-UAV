import cv2
import numpy as np

def bilinear_interpolation(img, x, y):
    """
    对给定的浮点坐标 (x, y) 执行双线性插值。
    """
    h, w = img.shape[:2]
    
    # 检查坐标是否在图像范围内
    if x < 0 or x >= w or y < 0 or y >= h:
        return 0
    
    # 获取四个邻近像素的坐标
    x0 = int(np.floor(x))
    x1 = min(x0 + 1, w - 1)
    y0 = int(np.floor(y))
    y1 = min(y0 + 1, h - 1)
    
    # 计算坐标差
    dx = x - x0
    dy = y - y0
    
    # 双线性插值
    top_left = img[y0, x0]
    top_right = img[y0, x1]
    bottom_left = img[y1, x0]
    bottom_right = img[y1, x1]
    
    top = top_left * (1 - dx) + top_right * dx
    bottom = bottom_left * (1 - dx) + bottom_right * dx
    return top * (1 - dy) + bottom * dy

def warp_image(background, img, M, output_shape):
    """
    使用变换矩阵 M 对图像进行变形，并使用双线性插值。
    """
    h, w = output_shape
    warped_img = np.zeros((h, w, img.shape[2]), dtype=img.dtype)
    
    # 计算逆变换矩阵
    M_inv = np.linalg.inv(M)
    
    for y in range(h):  # 遍历目标图像的每一行
        for x in range(w):  # 遍历目标图像的每一列
            # 构造目标图像中的坐标向量
            dest_coord = np.array([x, y, 1])
            # 使用逆变换矩阵计算源图像中的坐标
            src_coord = M_inv @ dest_coord
            src_x = src_coord[0] / src_coord[2]
            src_y = src_coord[1] / src_coord[2]
            
            # 如果源坐标在源图像范围内，进行插值
            if 0 <= src_x < img.shape[1] and 0 <= src_y < img.shape[0]:
                for c in range(img.shape[2]):  # 遍历颜色通道
                    background[y, x, c] = bilinear_interpolation(img[:, :, c], src_x, src_y)
    
    return background

# 加载目标图像
image = cv2.imread('screen.jpg')

# 定义角点，顺序为左上、右上、右下、左下，坐标格式为 (x, y)
cap_corner = np.float32([
    (0, 0),
    (image.shape[1]-1, 0),
    (image.shape[1]-1, image.shape[0]-1),
    (0, image.shape[0]-1)
])

# 这是您在屏幕上点击得到的角点，需要根据实际情况调整
img_corner = np.float32([
    (414, 864),    # 左上
    (1634, 218),   # 右上
    (1647, 1254),  # 右下
    (332, 1423)    # 左下
])

# 计算透视变换矩阵
M = cv2.getPerspectiveTransform(cap_corner, img_corner)

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        raise Exception("No camera detected!")

    # 将摄像头捕获的帧调整为目标图像的尺寸
    frame_resized = cv2.resize(frame, (image.shape[1], image.shape[0]))

    # 使用自定义的 warp_image 函数对帧进行透视变换
    new_frame = warp_image(image, frame_resized, M, (image.shape[0], image.shape[1]))

    # # 创建掩码，标记变形后图像中非零像素的位置
    # mask = np.any(new_frame != [0, 0, 0], axis=2).astype(np.uint8) * 255

    # # 创建反向掩码，标记原始图像中需要保留的部分
    # mask_inv = cv2.bitwise_not(mask)

    # # 提取原始图像的背景部分
    # img_bg = cv2.bitwise_and(image, image, mask=mask_inv)

    # # 提取变形后图像的前景部分
    # img_fg = cv2.bitwise_and(new_frame.astype(np.uint8), new_frame.astype(np.uint8), mask=mask)
    # cv2.imwrite("fg.jpg", img_fg)

    # # 合并背景和前景，得到最终的叠加图像
    # combined = cv2.add(img_bg, img_fg)

    # 显示结果
    cv2.imwrite("warp.jpg", new_frame)
