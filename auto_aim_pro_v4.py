import signal
import sys
import warnings
from collections import deque
from multiprocessing import Pipe, Process, Array
from multiprocessing.dummy import Lock
from statistics import median
from time import time

import cv2
import mss
import numpy as np
import pynput
import torch
import win32con
import win32gui
from pynput.mouse import Button

# 创建一个命名窗口
# loadConfig
# 消除警告信息
from simple_pid import PID

from auto_scripts.configs import MONITOR, CONF_THRES, IOU_THRES, LINE_THICKNESS, SHOW_LABEL, SHOW_IMG, SCREEN_NAME, \
    RESIZE_X, RESIZE_Y, LOCK_X, LOCK_Y, IMGSZ
from auto_scripts.get_model import load_model_infos
from auto_scripts.grabscreen import grab_screen_v2
from auto_scripts.mouse_controller_v2 import lock_v3
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_coords, xyxy2xywh
from utils.plots import Annotator, colors

warnings.filterwarnings('ignore')


# 启用 mss 截图
sct = mss.mss()

# 锁开关
LOCK_MOUSE = False

# 鼠标控制
mouse = pynput.mouse.Controller()

# 进程间共享数组
arr = Array('d', 4)
arr[2] = 0  # pid控制最新时间
arr[3] = 1  # 实时 fps

# 进程锁
process_lock = Lock()


# 点击监听
def on_click(x, y, button, pressed):
    global LOCK_MOUSE
    if pressed and button == Button.right:
        LOCK_MOUSE = not LOCK_MOUSE
        print('LOCK_MOUSE', LOCK_MOUSE)


def img_init(p1):
    print('进程 img_init 启动 ...')
    # loadModel
    model, device, half = load_model_infos()

    # 获取模型其他参
    stride = int(model.stride.max())  # model stride
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names


    while True:
        # --- 图像变化 ---
        # 获取指定位置 MONITOR 大小
        # img0 = sct.grab(MONITOR)
        # img0 = np.array(img0)
        # # 将图片转 BGR
        # img0 = cv2.cvtColor(img0, cv2.COLOR_BGRA2BGR)
        #
        # # 将图片缩小指定大小
        # img0 = cv2.resize(img0, (SCREEN_WIDTH, SCREEN_HEIGHT))
        img0 = grab_screen_v2(region=tuple(MONITOR.values()))

        # Padded resize
        img = letterbox(img0, IMGSZ, stride=stride)[0]

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()

        # 归一化处理
        img = img / 255.
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim

        pred = model(img, augment=False, visualize=False)[0]
        # NMS
        pred = non_max_suppression(pred, CONF_THRES, IOU_THRES, None, False, max_det=1000)
        aims = []
        for i, det in enumerate(pred):
            s = ''
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            # 设置方框绘制
            annotator = Annotator(img0, line_width=LINE_THICKNESS, example=str(names))
            if len(det):
                #  Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    # 获取类别索引
                    c = int(cls)  # integer class
                    # bbox 中的坐标
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh

                    line = (c, *xywh)  # label format
                    aims.append(line)
                    # 图片绘制
                    label = SHOW_LABEL and names[c] or None
                    annotator.box_label(xyxy, label, color=colors(c, True))
        p1.send((img0, aims))


def img_show(c1, p2, arr):
    print('进程 img_show 启动 ...')
    show_up = False
    show_tips = True
    signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))
    font = cv2.FONT_HERSHEY_SIMPLEX
    while True:
        # 展示窗口
        img0, aims = c1.recv()
        p2.send(aims)
        if show_tips:
            print('传输坐标中 ...')
        if SHOW_IMG:
            fps = f'{arr[3]:.2f}'
            cv2.putText(img0, fps, (50, 50), font, 1.2, (0, 255, 0), LINE_THICKNESS * 2)
            cv2.namedWindow(SCREEN_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(SCREEN_NAME, RESIZE_X, RESIZE_Y)
            cv2.imshow(SCREEN_NAME, img0)
            # 重设窗口大小
            if not show_up:
                hwnd = win32gui.FindWindow(None, SCREEN_NAME)
                win32gui.ShowWindow(hwnd, win32con.SW_SHOWNORMAL)
                win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0,
                                      win32con.SWP_NOMOVE | win32con.SWP_NOACTIVATE | win32con.SWP_NOOWNERZORDER |
                                      win32con.SWP_SHOWWINDOW | win32con.SWP_NOSIZE)
                show_up = not show_up
            k = cv2.waitKey(1)
            if k % 256 == 27:
                cv2.destroyAllWindows()
                p2.send('exit')
                exit('结束 img_show 进程中 ...')
        if show_tips:
            show_tips = False


# 加锁换值
def change_withlock(arrays, var, target_var, locker):
    with locker:
        arrays[var] = target_var

def get_bbox(c2, arr):
    global LOCK_MOUSE
    print('进程 get_bbox 启动 ...')
    # print(f'飞易来/文盒驱动加载状态: {msdkok}')
    # print(f'gmok加载状态: {gmok}')
    # ...or, in a non-blocking fashion:
    listener = pynput.mouse.Listener(on_click=on_click)
    listener.start()

    # 初始化一个事件队列
    process_times = deque()

    # 初始化 pid
    pid_x = PID(0.15, 0.0, 0.0, setpoint=0, sample_time=0.006, )
    pid_y = PID(0.15, 0.0, 0.0, setpoint=0, sample_time=0.006, )
    SMALL_FLOAT = np.finfo(np.float64).eps  # 初始化一个尽可能小却小得不过分的数
    while True:
        aims = c2.recv()
        # 花费时间
        last_time = arr[2]
        # 实时 fps
        fps = arr[3]
        if isinstance(aims, str):
            exit('结束 get_bbox 进程中 ...')
        else:
            if aims and LOCK_MOUSE:
                lock_v3(aims, mouse, LOCK_X, LOCK_Y)
        current_time = time()
        # 耗费时间
        time_used = current_time - last_time

        # 更新时间
        change_withlock(arr, 2, current_time, process_lock)
        process_times.append(time_used)
        median_time = median(process_times)
        pid_x.sample_time = pid_y.sample_time = median_time
        pid_x.kp = pid_y.kp = 1 / pow(fps / 3, 1 / 3)
        # 更新 fps
        change_withlock(arr, 3, 1 / median_time if median_time > 0 else 1 / (median_time + SMALL_FLOAT), process_lock)


if __name__ == '__main__':
    # 父进程创建Queue，并传给各个子进程：
    p1, c1 = Pipe()
    p2, c2 = Pipe()
    reader1 = Process(target=get_bbox, args=(c2, arr))
    reader2 = Process(target=img_show, args=(c1, p2, arr))
    writer = Process(target=img_init, args=(p1,))
    # 启动子进程 reader，读取:
    reader1.start()
    reader2.start()
    # 启动子进程 writer，写入:
    writer.start()

    # 等待 reader 结束:
    reader1.join()
    reader2.join()
    # 等待 writer 结束:
    writer.terminate()
    writer.join()
    exit('结束 img_init 进程中 ...')
