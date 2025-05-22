from ultralytics import YOLO

# 加载预训练模型
model = YOLO("yolov5su.pt")

# 测试图片推理
results = model("https://ultralytics.com/images/bus.jpg")

# 遍历结果并显示
for result in results:
    result.show()  # 显示每张图片的结果

