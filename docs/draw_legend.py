import cv2  
import numpy as np  
  
# 图片的尺寸  
width = 855  
height = 115  
  
# 创建一个空白图片，背景色为白色  
image = np.ones((height, width, 3), dtype=np.uint8) * 255  
  
# 定义矩形框的颜色（BGR格式）  
colors = [(0.95, 0.95, 0.95), (0.6, 0.6, 0.6), (0.9400000000000001, 0.7818, 0.66), (0.9400000000000001, 0.8868, 0.66), (0.8882000000000001, 0.9400000000000001, 0.66), (0.7832000000000001, 0.9400000000000001, 0.66), (0.6782000000000001, 0.9400000000000001, 0.66), (0.66, 0.9400000000000001, 0.7468000000000001),
          (0.66, 0.9400000000000001, 0.8518000000000001), (0.66, 0.9232, 0.9400000000000001), (0.66, 0.8182, 0.9400000000000001), (0.66, 0.7132, 0.9400000000000001), (0.7117999999999999, 0.66, 0.9400000000000001), (0.8168, 0.66, 0.9400000000000001),
          (0.9218, 0.66, 0.9400000000000001), (0.9400000000000001, 0.66, 0.8531999999999998), (0.9400000000000001, 0.66, 0.748199999999999), (1, 1, 0), (0, 1, 1), (1, 0, 1),
          (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (0, 1, 1), (1, 0, 1)
          ]  
  
# 定义动物名称  
# animal_names = ["Nav_area", "obstacle","chair","couch", "plant", "bed", "toilet", "tv",  
#                 "table", "picture", "sink", "appliances", "objects", "fireplace",  
#                 "lighting", "cabinet","stairs", "Squirrel", "Bear", "Ant",  
#                 "Penguin", "Kangaroo", "Whale", "Turtle"]  
animal_names = ["Nav_area", "obstacle","chair","couch", "plant", "bed", "toilet", "tv",  
                "bathtub", "shower", "fireplace", "appliances", "towel", "sink",  
                "chest_of_drawers", "table","stairs", "Squirrel", "Bear", "Ant",  
                "Penguin", "Kangaroo", "Whale", "Turtle"]    


# 每个矩形框的大小  
rect_width = (width - 300) // 12  # 留出一些空间用于显示动物名称和间距  
rect_height = height // 8
  
# 矩形框和文本之间的间距  
spacing = (width - 6 * rect_width) // 7  
vertical_spacing = (height - 4 * rect_height) // 5 
text_offset = 3  # 文本距离矩形框的水平偏移量  
  
# 绘制矩形框和添加动物名称  
for i in range(6):  
    for j in range(4):  
        index = i * 4 + j  
          
        # 检查索引是否超出动物名称列表  
        if index < len(animal_names) - 7:  
            x = i * (rect_width + spacing) + spacing // 2  
            y = j * (rect_height + vertical_spacing) + vertical_spacing //2  # 增加一些垂直间距  
              
            # 绘制矩形框   tuple(map(lambda x: int(x * 255), colors[index]))  
            # B = tuple(map(lambda x: int(x * 255), colors[index]))  
            # A = tuple(map(int, colors[index] * 255))
            cv2.rectangle(image, (x, y), (x + rect_width, y + rect_height), tuple(map(lambda x: int(x * 255), (colors[index][2],colors[index][1],colors[index][0]))), -1)  # -1 表示填充矩形  
              
            # 添加动物名称  
            text_x = x + rect_width + text_offset  
            text_y = y + rect_height // 2 + text_offset  # 文本垂直居中  
            cv2.putText(image, animal_names[index], (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

# 保存图片  

cv2.imwrite("output.png", image)  