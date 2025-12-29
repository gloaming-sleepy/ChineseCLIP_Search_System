# 路径修改总结

## ✅ 已修改的文件

所有硬编码的绝对路径已经改为相对路径，使用 `os.path.join()` 和 `PROJECT_ROOT` 变量。

### 1. **utils.py** ✅
- **修改前**:
  - `CLIP_CODE_PATH = r"D:\run\CLIP_models\Chinese-CLIP"`
  - `IMG_TSV = r"D:\run\CLIP_models\Text2Image-Retrieval-main\datapath\datasets\Flickr30k-CN\test_imgs.tsv"`
  - `sys.path.append(r"D:\run\CLIP_models\Chinese-CLIP")`
  - `MODEL_PATH = r"D:\run\CLIP_models\Text2Image-Retrieval-main\datapath\experiments\..."`

- **修改后**:
  ```python
  PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
  CLIP_CODE_PATH = os.path.join(PROJECT_ROOT, "Chinese-CLIP")
  IMG_TSV = os.path.join(PROJECT_ROOT, "datapath", "datasets", "Flickr30k-CN", "test_imgs.tsv")
  MODEL_PATH = os.path.join(PROJECT_ROOT, "datapath", "experiments",
                            "flickr30k_finetune_pycharm", "checkpoints", "epoch_latest.pt")
  ```

### 2. **build_db.py** ✅
- **修改前**: 3个绝对路径
- **修改后**: 全部使用 `PROJECT_ROOT` + `os.path.join()`
  ```python
  PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
  CLIP_CODE_PATH = os.path.join(PROJECT_ROOT, "Chinese-CLIP")
  MODEL_PATH = os.path.join(PROJECT_ROOT, "datapath", "experiments", ...)
  IMAGE_DATA = os.path.join(PROJECT_ROOT, "datapath", "datasets", "Flickr30k-CN", "test_imgs.tsv")
  OUTPUT_FILE = os.path.join(PROJECT_ROOT, "image_features.json")
  ```

### 3. **export_onnx.py** ✅
- **修改前**: 3个绝对路径
- **修改后**: 全部使用相对路径
  ```python
  PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
  CLIP_CODE_PATH = os.path.join(PROJECT_ROOT, "Chinese-CLIP")
  MODEL_PATH = os.path.join(PROJECT_ROOT, "datapath", "experiments", ...)
  OUTPUT_FILE = os.path.join(PROJECT_ROOT, "vit-b-16-text.onnx")
  ```

### 4. **plot_log.py** ✅
- **修改前**: `LOG_FILE_PATH = r"D:\run\CLIP_models\..."`
- **修改后**:
  ```python
  PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
  LOG_FILE_PATH = os.path.join(PROJECT_ROOT, "datapath", "experiments",
                               "flickr30k_finetune_pycharm", "out_2025-11-27-15-03-18.log")
  ```

### 5. **run_finetune.py** ✅
- **修改前**: 多个字符串拼接的绝对路径
- **修改后**:
  ```python
  PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
  ROOT_PATH = os.path.join(PROJECT_ROOT, "datapath")
  CHINESE_CLIP_PATH = os.path.join(PROJECT_ROOT, "Chinese-CLIP")
  MAIN_SCRIPT = os.path.join(CHINESE_CLIP_PATH, "cn_clip", "training", "main.py")

  config = {
      "--train-data": os.path.join(ROOT_PATH, "datasets", "Flickr30k-CN", "lmdb", "train"),
      "--val-data": os.path.join(ROOT_PATH, "datasets", "Flickr30k-CN", "lmdb", "valid"),
      ...
  }
  ```

## 📝 其他文件（不需要修改）

- **训练日志文件** (`.log`, `params_*.txt`): 这些是训练时生成的日志，包含历史路径信息，不影响运行
- **UPLOAD_GUIDE.md**: 仅包含示例命令，已更新

## ✨ 优点

1. ✅ **跨平台兼容**: Windows、Linux、macOS 都能运行
2. ✅ **可移植性**: 项目可以放在任何目录下
3. ✅ **团队协作**: 其他人 clone 后不需要修改路径
4. ✅ **GitHub 友好**: 上传后其他用户可以直接使用

## 🎯 使用方法

现在用户只需要：

```bash
# 1. Clone 项目
git clone https://github.com/your-username/Text2Image-Retrieval.git
cd Text2Image-Retrieval

# 2. 安装依赖和 Chinese-CLIP
git clone https://github.com/OFA-Sys/Chinese-CLIP.git
cd Chinese-CLIP
pip install -e .
cd ..

# 3. 直接运行（路径自动识别）
python app.py
python build_db.py
python export_onnx.py
```

无需修改任何路径配置！🎉

## 🔍 路径逻辑说明

所有脚本使用统一的路径获取方式：

```python
# 获取当前脚本所在的目录（项目根目录）
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 基于 PROJECT_ROOT 构建所有路径
SOME_PATH = os.path.join(PROJECT_ROOT, "datapath", "subfolder", "file.txt")
```

这样无论项目放在哪里，都能正确找到文件。
