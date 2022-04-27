import sys

from PyQt5.QtWidgets import QApplication

app = QApplication(sys.argv)
_desktop = QApplication.desktop()
# 获取显示器分辨率大小
_screen_rect = _desktop.screenGeometry()
SCREEN_WIDTH = _screen_rect.width()
SCREEN_HEIGHT = _screen_rect.height()

# 实时显示窗口名称
SCREEN_NAME = 'Tanck'
# 游戏内分辨率大小
GAME_X, GAME_Y = (1928, 1080)
# GAME_X, GAME_Y = (SCREEN_WIDTH, SCREEN_HEIGHT)

# 重设窗口大小
RESIZE_X = 388
RESIZE_Y = 488

# mss 截图指定区域
MONITOR = {"top": int(GAME_X / 2 - RESIZE_X / 2), "left": int(GAME_Y / 2 - RESIZE_Y / 2), "width": RESIZE_X,
           "height": RESIZE_Y}

# 模型文件
WEIGHTS = r'C:\Users\Administrator\Desktop\yolo\aimcf_yolov5\cf_best.pt'

# 预测转换图片大小
IMGSZ = (640, 640)

# 置信度
CONF_THRES = .25

# IOU
IOU_THRES = .45

# 方框宽度
LINE_THICKNESS = 4

# 是否显示图像
SHOW_IMG = True

# 是否显示 label
SHOW_LABEL = False
