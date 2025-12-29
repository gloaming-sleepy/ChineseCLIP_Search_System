import os
import sys
import numpy as np
import onnxruntime
import torch
import json
import pandas as pd
import base64
from PIL import Image
from io import BytesIO

# ================= 配置 =================
# 获取项目根目录（utils.py 所在目录）
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 1. 源码路径（Chinese-CLIP 应该在项目根目录下）
CLIP_CODE_PATH = os.path.join(PROJECT_ROOT, "Chinese-CLIP")
if os.path.exists(CLIP_CODE_PATH):
    sys.path.append(CLIP_CODE_PATH)
import cn_clip.clip as clip

# 2. 文件路径（使用相对路径）
ONNX_MODEL = os.path.join(PROJECT_ROOT, "vit-b-16-text.onnx")  # ONNX 模型
FEAT_JSON = os.path.join(PROJECT_ROOT, "image_features.json")  # 图像特征库
# 测试集 TSV (用来显示图片)
IMG_TSV = os.path.join(PROJECT_ROOT, "datapath", "datasets", "Flickr30k-CN", "test_imgs.tsv")

# 3. 定义模型名字 (text2image.py里用的)
clip_base = "中文CLIP(Base)"
yes = "是"
no = "否"

# =======================================

print("1. 正在加载 ONNX 模型...")
# 优先使用 GPU 推理，如果报错会自动切回 CPU
sess_options = onnxruntime.SessionOptions()
try:
    session = onnxruntime.InferenceSession(ONNX_MODEL, sess_options,
                                           providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
except:
    session = onnxruntime.InferenceSession(ONNX_MODEL, sess_options, providers=["CPUExecutionProvider"])

print("2. 正在加载图片特征库...")
img_ids = []
img_feats = []
with open(FEAT_JSON, "r") as f:
    for line in f:
        obj = json.loads(line.strip())
        img_ids.append(obj['image_id'])
        img_feats.append(obj['feature'])

# 转为矩阵: [Num_Images, Feature_Dim]
img_feats_matrix = np.array(img_feats, dtype=np.float32)
# L2 归一化，使相似度分数范围在 [0, 1]
img_feats_matrix = img_feats_matrix / np.linalg.norm(img_feats_matrix, axis=1, keepdims=True)
# 转置为 [Feature_Dim, Num_Images] 以便矩阵乘法
img_feats_matrix = img_feats_matrix.T
print(f"库加载完成，共 {len(img_ids)} 张图片")

print("3. 正在加载图片索引 (用于显示)...")
df_imgs = pd.read_csv(IMG_TSV, sep='\t', header=None, names=['id', 'b64'])
df_imgs.set_index('id', inplace=True)


def decode_b64(b64_str):
    return Image.open(BytesIO(base64.b64decode(b64_str))).convert('RGB')


# ================= 图搜图功能 =================
# 需要加载完整模型用于图像编码
from cn_clip.clip import load_from_name

# 全局变量：延迟加载模型
_image_model = None
_image_preprocess = None

def get_image_model():
    """延迟加载图像编码模型"""
    global _image_model, _image_preprocess
    if _image_model is None:
        print("正在加载图像编码模型...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _image_model, _image_preprocess = load_from_name("ViT-B-16", device=device, download_root='./')

        # 加载微调权重（使用相对路径）
        MODEL_PATH = os.path.join(PROJECT_ROOT, "datapath", "experiments",
                                  "flickr30k_finetune_pycharm", "checkpoints", "epoch_latest.pt")
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        sd = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        if next(iter(sd.items()))[0].startswith('module.'):
            sd = {k[7:]: v for k, v in sd.items()}
        _image_model.load_state_dict(sd)
        _image_model.eval()
        print("✅ 图像编码模型加载完成")
    return _image_model, _image_preprocess

def encode_query_image(query_img):
    """
    对上传的查询图像进行编码
    """
    model, preprocess = get_image_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with torch.no_grad():
        # 预处理图像
        image_tensor = preprocess(query_img).unsqueeze(0).to(device)
        # 编码
        feat = model.encode_image(image_tensor)
        # 归一化
        feat /= feat.norm(dim=-1, keepdim=True)

    return feat.cpu().numpy()

def image_search_api(query_img, return_n, model_name, thumbnail):
    """
    图搜图 API（对应文搜图的 clip_api）
    Args:
        query_img: 用户上传的 PIL Image
        return_n: 返回结果数量
        model_name: 模型名（占位，保持接口一致）
        thumbnail: 是否缩略图（占位）
    Returns:
        [(PIL.Image, str), ...] Gradio Gallery 格式
    """
    return_n = int(return_n)

    # 1. 对查询图像编码
    query_feat = encode_query_image(query_img)  # [1, 512]

    # 2. 计算与库中所有图像的相似度
    logits = query_feat @ img_feats_matrix  # [1, N]
    probs = logits[0]

    # 3. 取 Top K（排除自身，如果查询图在库中的话）
    top_indices = np.argsort(probs)[-(return_n+5):][::-1]  # 多取几个以防自身

    results = []
    for idx in top_indices:
        if len(results) >= return_n:
            break

        score = probs[idx]
        img_id = img_ids[idx]

        # 获取原图
        try:
            row = df_imgs.loc[img_id]
            b64 = row['b64']
            if not isinstance(b64, str):
                b64 = b64.iloc[0]

            img = decode_b64(b64)
            results.append((img, f"ID:{img_id} 相似度:{score:.3f}"))
        except:
            continue

    return results
# ============================================


# Gradio 调用的主函数
def clip_api(text, return_n, model_name, thumbnail):
    return_n = int(return_n)

    # 1. 文本转 Token 
    text_input = clip.tokenize([text]).numpy()

    # 2. ONNX 推理文本特征
    text_feat = session.run(["unnorm_text_features"], {"text": text_input})[0]

    # 3. 归一化
    text_feat = text_feat / np.linalg.norm(text_feat, axis=1, keepdims=True)

    # 4. 计算相似度: 
    logits = text_feat @ img_feats_matrix
    probs = logits[0]

    # 5. 取 Top K
    top_indices = np.argsort(probs)[-return_n:][::-1]

    # 6. 返回结果
    results = []
    for idx in top_indices:
        score = probs[idx]
        img_id = img_ids[idx]

        # 获取原图
        try:
            row = df_imgs.loc[img_id]
            b64 = row['b64']
            if not isinstance(b64, str): b64 = b64.iloc[0]  # 处理重复ID

            img = decode_b64(b64)
            results.append((img, f"ID:{img_id} 分数:{score:.3f}"))
        except:
            continue

    return results