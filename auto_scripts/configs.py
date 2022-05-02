import sys

import win32gui
from PyQt5.QtWidgets import QApplication

app = QApplication(sys.argv)
_desktop = QApplication.desktop()
# 获取显示器分辨率大小
_screen_rect = _desktop.screenGeometry()
SCREEN_WIDTH = _screen_rect.width()
SCREEN_HEIGHT = _screen_rect.height()

hwnd = win32gui.FindWindow(None, '穿越火线')
rect = win32gui.GetWindowRect(hwnd)
w = rect[2] - rect[0]
h = rect[3] - rect[1]
print(f'w:{w}, h:{h}')
# 实时显示窗口名称
SCREEN_NAME = 'Tanck'
# 游戏内分辨率大小
GAME_X, GAME_Y = (1928, 1080)
# GAME_X, GAME_Y = (SCREEN_WIDTH, SCREEN_HEIGHT)

# 重设窗口大小
RESIZE_X = 388
RESIZE_Y = 488

# 截图范围
LOCK_WIDTH = 388
LOCK_HEIGHT = 388

# 截图的左上角坐标
LOCK_Y = h / 2 - LOCK_HEIGHT / 2 + rect[1]
LOCK_X = w / 2 - LOCK_WIDTH / 2 + rect[0]

HEAD_OFFSET = 0 #雷神 20 ， AK 15

# mss 截图指定区域
MONITOR = {"top": int(LOCK_X), "left": int(LOCK_Y), "width": LOCK_WIDTH,
           "height": LOCK_HEIGHT}

# 模型文件
WEIGHTS = r'C:\Users\Administrator\Desktop\yolo\aimcf_yolov5\best.pt'

# 预测转换图片大小
IMGSZ = (640, 640)

# 置信度
CONF_THRES = .42

# IOU
IOU_THRES = .45

# 方框宽度
LINE_THICKNESS = 4

# 是否显示图像
SHOW_IMG = True

# 是否显示 label
SHOW_LABEL = False
