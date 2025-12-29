"""
图搜图界面 - Image-to-Image Search
基于 Chinese-CLIP 微调模型实现以图搜图功能
"""
import gradio as gr
from utils import image_search_api, clip_base, yes, no

# 定义页面描述文案
description = "本项目基于 Chinese-CLIP 微调后的图搜图检索系统 Demo，支持上传图片检索本地 Flickr30k-CN 测试集中的图片。"


def image2image_gr():
    """构建图搜图界面"""

    title = "<h1 align='center'>🔍 中文CLIP图搜图应用 (微调版)</h1>"

    with gr.Blocks() as demo:
        gr.Markdown(title)

        # 顶部说明区域
        with gr.Row():
            gr.Markdown("""
            ## 🖼️ 图搜图检索系统

            上传一张查询图片，系统会自动在 Flickr30k-CN 测试集中找到视觉相似的图像。
            """)

        with gr.Row():
            with gr.Column(scale=1):
                # 图像上传控件
                query_image = gr.Image(
                    label="📤 上传查询图片",
                    type="pil",
                    elem_id="query_img"
                )

                # 返回数量滑块
                num = gr.Slider(
                    minimum=1,
                    maximum=20,
                    step=1,
                    value=8,
                    label="返回图片数量",
                    elem_id=2
                )

                # 模型选择
                model = gr.Radio(
                    label="模型选择",
                    choices=[clip_base],
                    value=clip_base,
                    elem_id=3
                )

                # 缩略图选项
                thumbnail = gr.Radio(
                    label="是否返回缩略图",
                    choices=[yes, no],
                    value=yes,
                    elem_id=4
                )

                # 搜索按钮（橙色风格）
                btn = gr.Button("🔍 搜索相似图片", variant="primary", elem_id="search_btn")

                # 提示信息
                gr.Markdown("""
                **提示：**
                - 支持常见图片格式（JPG, PNG, BMP等）
                - 首次使用需加载模型，约需10-30秒
                - 建议上传清晰、主体明确的图片以获得更好效果
                """)

            with gr.Column(scale=4):
                # 结果展示画廊
                out = gr.Gallery(
                    label="🎯 相似图片检索结果（按相似度从高到低排序）",
                    columns=4,
                    height=600
                )

        # 输入参数列表
        inputs = [query_image, num, model, thumbnail]

        # 绑定点击事件
        btn.click(fn=image_search_api, inputs=inputs, outputs=out)

    return demo


if __name__ == "__main__":
    demo = image2image_gr()
    demo.queue().launch(
        server_name="127.0.0.1",
        server_port=7861,
        share=False
    )
