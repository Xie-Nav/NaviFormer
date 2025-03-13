import cv2
import numpy as np


def get_contour_points(pos, origin, size=20):
    x, y, o = pos
    pt1 = (int(x) + origin[0],
           int(y) + origin[1])
    pt2 = (int(x + size / 1.5 * np.cos(o + np.pi * 4 / 3)) + origin[0],
           int(y + size / 1.5 * np.sin(o + np.pi * 4 / 3)) + origin[1])
    pt3 = (int(x + size * np.cos(o)) + origin[0],
           int(y + size * np.sin(o)) + origin[1])
    pt4 = (int(x + size / 1.5 * np.cos(o - np.pi * 4 / 3)) + origin[0],
           int(y + size / 1.5 * np.sin(o - np.pi * 4 / 3)) + origin[1])

    return np.array([pt1, pt2, pt3, pt4])


def draw_line(start, end, mat, steps=25, w=1):
    for i in range(steps + 1):
        x = int(np.rint(start[0] + (end[0] - start[0]) * i / steps))
        y = int(np.rint(start[1] + (end[1] - start[1]) * i / steps))
        mat[x - w:x + w, y - w:y + w] = 1
    return mat


def init_vis_image(goal_name, legend):
    # goal_name 当前目标对应的字符；legend = none； 
    # vis_image是我们可视化的一个视频的大小，也就是(655, 1165, 3)
    vis_image = np.ones((656, 1168, 3)).astype(np.uint8) * 255
    # font = 0  
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (20, 20, 20)  # BGR
    thickness = 2

    text = "Observations (Goal: {})".format(goal_name)
       # 需要在图像上添加文本时（例如标注、标题等），cv2.getTextSize这个函数可以帮助你确定文本的大小和位置。
       #  text = "Observations (Goal: '目标物体字符')"  font = 0；fontScale = 1；thickness = 2
       #   font  字体类型；fontScale 字体缩放因子。这个值越大，字体越大。；thickness 文本的线条厚度。
       # 第一个元素是文本的宽度和高度（以像素为单位），第二个元素是基线到文本底部的距离
       # textsize 是文本的宽度和高度（以像素为单位）当前是(394, 22)
    textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
       #   textX 和 textY是我们的text在vis_image中的左下角的坐标
       # 这个"Observations (Goal: '目标物体字符')"在vis_image在的左上角
    textX = (640 - textsize[0]) // 2 + 15
    textY = (50 + textsize[1]) // 2
    vis_image = cv2.putText(vis_image, text, (textX, textY),
                            font, fontScale, color, thickness,
                            cv2.LINE_AA)

    text = "Predicted Semantic Map"
    textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
    textX = 640 + (480 - textsize[0]) // 2 + 30
    textY = (50 + textsize[1]) // 2
       # 这个  "Predicted Semantic Map" 在 vis_image右上角
    vis_image = cv2.putText(vis_image, text, (textX, textY),
                            font, fontScale, color, thickness,
                            cv2.LINE_AA)

    # draw outlines
    color = [100, 100, 100]
       # 第一个框是为了放每一帧的第一视角图片 
    vis_image[49, 15:655] = color
    vis_image[50:530, 14] = color
    vis_image[530, 15:655] = color
    vis_image[50:530, 655] = color
    
    #  第二个框 这个就是我们的local BEVmap的一个可视化图大小480 * 480  
    vis_image[49, 670:1150] = color
    vis_image[50:530, 669] = color
    vis_image[50:530, 1150] = color
    vis_image[530, 670:1150] = color

    # draw legend
       #  这是为了画我们的图例   
    lx, ly, _ = legend.shape
#     将我们的图例直接放进去，应该是物体的图例
    vis_image[537:537 + lx, 155:155 + ly, :] = legend

    return vis_image
