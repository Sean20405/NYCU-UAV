import cv2

# 回調函數，當鼠標點擊時獲取點的坐標
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # 標記左鍵單擊的點
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Image", img)

# 加載圖像
img = cv2.imread('screen.jpg')

# 創建窗口並設置鼠標回調
cv2.imshow("Image", img)
cv2.setMouseCallback("Image", click_event)

cv2.waitKey(0)
cv2.destroyAllWindows()
