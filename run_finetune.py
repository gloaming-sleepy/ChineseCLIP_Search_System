import os
import sys

# ================= 配置区域（使用相对路径）=================
# 获取脚本所在目录（项目根目录）
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.join(PROJECT_ROOT, "datapath")
CHINESE_CLIP_PATH = os.path.join(PROJECT_ROOT, "Chinese-CLIP")
MAIN_SCRIPT = os.path.join(CHINESE_CLIP_PATH, "cn_clip", "training", "main.py")
# ===========================================

config = {
    # 核心路径
    "--train-data": os.path.join(ROOT_PATH, "datasets", "Flickr30k-CN", "lmdb", "train"),
    "--val-data": os.path.join(ROOT_PATH, "datasets", "Flickr30k-CN", "lmdb", "valid"),
    "--resume": os.path.join(ROOT_PATH, "pretrained_weights", "clip_cn_vit-b-16.pt"),
    "--logs": os.path.join(ROOT_PATH, "experiments"),
    "--name": "flickr30k_finetune_pycharm",

    # 【重点修改】强制单进程加载，解决 Windows 报错
    "--num-workers": "0",
    "--valid-num-workers": "0",

    # 显存与训练参数
    "--batch-size": "32",
    "--max-epochs": "3",
    "--lr": "5e-5",
    "--vision-model": "ViT-B-16",
    "--text-model": "RoBERTa-wwm-ext-base-chinese",
    "--context-length": "52",
    "--warmup": "100",
    "--accum-freq": "1",
    "--wd": "0.001",

    # 开关
    "--reset-data-offset": True,
    "--reset-optimizer": True,
    "--report-training-batch-acc": True,
    "--use-augment": True,
}


def run():
    # 设置环境变量
    os.environ["DATAPATH"] = ROOT_PATH
    os.environ["PYTHONPATH"] = os.environ.get("PYTHONPATH", "") + os.pathsep + os.path.join(PROJECT_ROOT, "cn_clip")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "8514"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"

    # 构造命令
    cmd = [
        sys.executable, "-m", "torch.distributed.launch",
        "--nproc_per_node=1",
        "--use_env",
        MAIN_SCRIPT
    ]

    for key, val in config.items():
        if val is True:
            cmd.append(key)
        elif val is False:
            continue
        else:
            cmd.append(f"{key}={val}")

    full_cmd = " ".join(cmd)
    print(f"正在 PyCharm 中启动训练...\n命令: {full_cmd}\n")
    os.system(full_cmd)


if __name__ == '__main__':
    run()