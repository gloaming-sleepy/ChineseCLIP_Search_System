import re
import matplotlib.pyplot as plt
import os

# ================= 配置区域（使用相对路径）=================
# 获取脚本所在目录（项目根目录）
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
LOG_FILE_PATH = os.path.join(PROJECT_ROOT, "datapath", "experiments",
                             "flickr30k_finetune_pycharm", "out_2025-11-27-15-03-18.log")
# ===========================================

def parse_log(file_path):
    train_steps = []
    train_losses = []
    train_acc_i2t = []
    train_acc_t2i = []

    val_epochs = []
    val_losses = []
    val_acc_i2t = []
    val_acc_t2i = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 提取训练数据
            # Pattern: Global Steps: 10/... | Loss: 0.25... | Image2Text Acc: 96.88 | Text2Image Acc: 84.38
            if "Global Steps:" in line and "Loss:" in line:
                try:
                    step = int(re.search(r'Global Steps: (\d+)', line).group(1))
                    loss = float(re.search(r'Loss: ([\d\.]+)', line).group(1))
                    i2t = float(re.search(r'Image2Text Acc: ([\d\.]+)', line).group(1))
                    t2i = float(re.search(r'Text2Image Acc: ([\d\.]+)', line).group(1))

                    train_steps.append(step)
                    train_losses.append(loss)
                    train_acc_i2t.append(i2t)
                    train_acc_t2i.append(t2i)
                except:
                    continue

            # 提取验证数据
            # Pattern: Validation Result (epoch 1 ...) | Valid Loss: 0.54... | Image2Text Acc: 83.80 | ...
            if "Validation Result" in line:
                try:
                    epoch = int(re.search(r'epoch (\d+)', line).group(1))
                    v_loss = float(re.search(r'Valid Loss: ([\d\.]+)', line).group(1))
                    v_i2t = float(re.search(r'Image2Text Acc: ([\d\.]+)', line).group(1))
                    v_t2i = float(re.search(r'Text2Image Acc: ([\d\.]+)', line).group(1))

                    val_epochs.append(epoch)
                    val_losses.append(v_loss)
                    val_acc_i2t.append(v_i2t)
                    val_acc_t2i.append(v_t2i)
                except:
                    continue

    return (train_steps, train_losses, train_acc_i2t, train_acc_t2i), \
        (val_epochs, val_losses, val_acc_i2t, val_acc_t2i)


def plot_charts(train_data, val_data):
    t_steps, t_loss, t_i2t, t_t2i = train_data
    v_epochs, v_loss, v_i2t, v_t2i = val_data

    # 设置绘图风格
    plt.style.use('seaborn-v0_8-whitegrid')

    # 图1：训练 Loss 曲线 (展示收敛过程)
    plt.figure(figsize=(10, 5))
    plt.plot(t_steps, t_loss, label='Training Loss', color='#E24A33', alpha=0.6)
    plt.title('Training Loss Curve', fontsize=14)
    plt.xlabel('Global Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('report_loss_curve.png', dpi=300)
    print("✅ 已生成: report_loss_curve.png (Loss下降图)")
    plt.show()

    # 图2：训练准确率曲线
    plt.figure(figsize=(10, 5))
    plt.plot(t_steps, t_i2t, label='Image2Text Acc', color='#348ABD', alpha=0.5)
    plt.plot(t_steps, t_t2i, label='Text2Image Acc', color='#988ED5', alpha=0.5)
    plt.title('Training Accuracy Curve', fontsize=14)
    plt.xlabel('Global Steps')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.savefig('report_train_acc.png', dpi=300)
    print("✅ 已生成: report_train_acc.png (训练准确率图)")
    plt.show()

    # 图3：验证集准确率 (每个Epoch的结果)
    plt.figure(figsize=(8, 5))
    plt.plot(v_epochs, v_i2t, 'o-', label='Val Image2Text', linewidth=2)
    plt.plot(v_epochs, v_t2i, 's-', label='Val Text2Image', linewidth=2)
    plt.title('Validation Accuracy per Epoch', fontsize=14)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.xticks(v_epochs)  # 强制显示整数Epoch
    plt.legend()
    plt.grid(True)
    plt.savefig('report_val_result.png', dpi=300)
    print("✅ 已生成: report_val_result.png (验证集结果图)")
    plt.show()

    # 打印最终结果供复制
    print("\n" + "=" * 30)
    print("📝 最终验证集结果 (可直接复制到报告)")
    print("=" * 30)
    for i in range(len(v_epochs)):
        print(f"Epoch {v_epochs[i]}:")
        print(f"  - Valid Loss: {v_loss[i]:.4f}")
        print(f"  - Image2Text Acc: {v_i2t[i]:.2f}%")
        print(f"  - Text2Image Acc: {v_t2i[i]:.2f}%")
        print("-" * 20)


if __name__ == "__main__":
    # 检查路径是否存在
    import os

    if not os.path.exists(LOG_FILE_PATH):
        print("❌ 找不到 log 文件，请检查路径是否正确！")
    else:
        t_data, v_data = parse_log(LOG_FILE_PATH)
        if len(t_data[0]) == 0:
            print("❌ 读取到了文件，但没提取到数据。请确认 log 内容是否完整。")
        else:
            plot_charts(t_data, v_data)