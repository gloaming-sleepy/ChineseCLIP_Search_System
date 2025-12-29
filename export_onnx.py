import torch
import torch.onnx
import os
import sys

# ================= 配置路径（使用相对路径）=================
# 获取脚本所在目录（项目根目录）
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

CLIP_CODE_PATH = os.path.join(PROJECT_ROOT, "Chinese-CLIP")
MODEL_PATH = os.path.join(PROJECT_ROOT, "datapath", "experiments",
                          "flickr30k_finetune_pycharm", "checkpoints", "epoch_latest.pt")
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "vit-b-16-text.onnx")
# ===========================================

if os.path.exists(CLIP_CODE_PATH):
    sys.path.append(CLIP_CODE_PATH)
from cn_clip.clip import load_from_name


class TextEncoderWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.bert = model.bert
        self.text_projection = model.text_projection

    def forward(self, text):
        x = self.bert(text)
        # 回退到最简单的逻辑：
        # x[0] 是 BERT 输出序列 [batch, seq_len, 768]
        # 我们直接取第 0 个位置的 [CLS] 向量
        cls_feat = x[0][:, 0, :]
        return cls_feat @ self.text_projection


def export():
    device = "cpu"
    print(f"正在加载微调模型: {MODEL_PATH}")
    model, _ = load_from_name("ViT-B-16", device=device, download_root=PROJECT_ROOT)

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    sd = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    if next(iter(sd.items()))[0].startswith('module.'):
        sd = {k[7:]: v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()

    text_encoder = TextEncoderWrapper(model)
    dummy_input = torch.zeros((1, 52), dtype=torch.long)

    print("正在转换为 ONNX (无Pooler版)...")
    torch.onnx.export(
        text_encoder, dummy_input, OUTPUT_FILE,
        input_names=['text'], output_names=['unnorm_text_features'],
        dynamic_axes={'text': {0: 'batch_size'}, 'unnorm_text_features': {0: 'batch_size'}},
        opset_version=13
    )
    print(f"✅ 转换成功！文件保存至: {OUTPUT_FILE}")


if __name__ == '__main__':
    export()