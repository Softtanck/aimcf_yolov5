from auto_scripts.configs import LOCK_WIDTH, LOCK_HEIGHT, HEAD_OFFSET

def lock_v3(aims, mouse, x, y):
    from utils.now.mouse import mouse_xy
    mouse_pos_x, mouse_pos_y = mouse.position
    dist_list = []
    for _, x_c, y_c, _, _ in aims:
        dist = (LOCK_WIDTH * x_c + x - mouse_pos_x) ** 2 + (LOCK_HEIGHT * y_c + y - mouse_pos_y) ** 2
        dist_list.append(dist)

    # 获取当前离鼠标最近的
    det = aims[dist_list.index(min(dist_list))]
    tag, x_center, y_center, width, height = det
    x_center = LOCK_WIDTH * x_center + x
    y_center = LOCK_HEIGHT * y_center + y - HEAD_OFFSET
    # width = x * width
    height = LOCK_HEIGHT * height
    coef = 1
    # if tag == 1:
    #     offset_x = x_center - mouse_pos_x
    #     offset_y = y_center - mouse_pos_y - height / 8
    #     # offset_x *= coef
    #     mouse_xy(offset_x, offset_y)
    if tag == 0:
        offset_x = x_center - mouse_pos_x
        offset_y = y_center - mouse_pos_y
        offset_x *= coef
        mouse_xy(offset_x, offset_y)
    elif tag == 1:
        offset_x = x_center - mouse_pos_x
        offset_y = y_center - height / 4 - mouse_pos_y
        offset_x *= coef
        mouse_xy(offset_x, offset_y)