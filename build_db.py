import torch
import json
import base64
import pandas as pd
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import sys
import os

# ================= 配置路径（使用相对路径）=================
# 获取脚本所在目录（项目根目录）
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

CLIP_CODE_PATH = os.path.join(PROJECT_ROOT, "Chinese-CLIP")
MODEL_PATH = os.path.join(PROJECT_ROOT, "datapath", "experiments",
                          "flickr30k_finetune_pycharm", "checkpoints", "epoch_latest.pt")
IMAGE_DATA = os.path.join(PROJECT_ROOT, "datapath", "datasets", "Flickr30k-CN", "test_imgs.tsv")
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "image_features.json")
# ===========================================

if os.path.exists(CLIP_CODE_PATH):
    sys.path.append(CLIP_CODE_PATH)
from cn_clip.clip import load_from_name


def build():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("加载模型中...")
    model, preprocess = load_from_name("ViT-B-16", device=device, download_root='./')

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    sd = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    if next(iter(sd.items()))[0].startswith('module.'):
        sd = {k[7:]: v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()

    # 读取 TSV (假设没有表头，第一列ID，第二列Base64)
    df = pd.read_csv(IMAGE_DATA, sep='\t', header=None, names=['id', 'b64'])

    print(f"开始提取 {len(df)} 张图片的特征...")
    features = []

    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df)):
            try:
                img = Image.open(BytesIO(base64.b64decode(row['b64'])))
                image = preprocess(img).unsqueeze(0).to(device)
                feat = model.encode_image(image)
                feat /= feat.norm(dim=-1, keepdim=True)  # 归一化

                features.append({
                    "image_id": row['id'],
                    "feature": feat.cpu().numpy().tolist()[0]
                })
            except Exception as e:
                print(f"跳过坏图: {e}")

    # 保存
    with open(OUTPUT_FILE, "w") as f:
        for item in features:
            f.write(json.dumps(item) + "\n")
    print(f"✅ 特征库已生成: {OUTPUT_FILE}")


if __name__ == '__main__':
    build()