import os
import json
from anomalib.deploy import TorchInferencer

def generate_pseudo_labels(model, image_folder, threshold, output_json):
    model.eval()  # 确保模型处于推理模式
    pseudo_labels = {}
    inferencer = TorchInferencer(model, device='cuda')

    for image_name in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_name)

        if not image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
            continue  # 跳过非图片文件

        # 用你的模型定义的推理函数，得到异常分数
        anomaly_score = inferencer.predict(image_path)  # in: image(nmupy) -> out:

        # 根据阈值生成伪标签
        pseudo_label = 1 if anomaly_score > threshold else 0
        pseudo_labels[image_name] = pseudo_label

    # 保存伪标签到json文件
    with open(output_json, "w") as f:
        json.dump(pseudo_labels, f, indent=4)

    print(f"[✓] 伪标签已保存至: {output_json}")

