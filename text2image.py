"""
文搜图界面 - Text-to-Image Search
基于 Chinese-CLIP 微调模型实现文本检索图像功能
"""
import gradio as gr
from utils import clip_api, clip_base, yes, no

# 定义页面描述文案
description = "本项目为基于 Chinese-CLIP 微调后的图文检索系统 Demo。支持输入中文文本，实时检索本地 Flickr30k-CN 测试集中的图片。"


def text2image_gr():
    """构建文搜图界面"""

    # 示例查询
    examples = [
        ["游泳的狗", 8, clip_base, "是"],
        ["夜晚盛开的荷花", 8, clip_base, "是"],
        ["一个走在公园里的女孩", 8, clip_base, "是"],
        ["抱着孩子的男人", 8, clip_base, "是"]
    ]

    title = "<h1 align='center'>🔍 中文CLIP文到图搜索应用 (微调版)</h1>"

    with gr.Blocks() as demo:
        gr.Markdown(title)

        # 顶部说明区域
        with gr.Row():
            gr.Markdown("""
            ## 📝 文到图检索系统

            本项目为基于 Chinese-CLIP 微调后的图文检索系统 Demo。支持输入中文文本，实时检索本地 Flickr30k-CN 测试集中的图片。
            """)

        with gr.Row():
            with gr.Column(scale=1):
                # 文本输入框
                text = gr.Textbox(
                    value="骑自行车的人",
                    label="📝 请填写文本",
                    elem_id=0,
                    interactive=True
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
                btn = gr.Button("🔍 搜索", variant="primary", elem_id="search_btn")

            with gr.Column(scale=4):
                # 结果展示画廊
                out = gr.Gallery(
                    label="🎯 检索结果为（按相似度从高到低排序）：",
                    columns=4,
                    height=600
                )

        inputs = [text, num, model, thumbnail]

        # 绑定点击事件
        btn.click(fn=clip_api, inputs=inputs, outputs=out)

        # 绑定示例点击
        gr.Examples(
            examples,
            inputs=inputs,
            label="💡 Examples"
        )

    return demo


if __name__ == "__main__":
    with gr.TabbedInterface(
            [text2image_gr()],
            ["文到图搜索"],
    ) as demo:
        #旧：demo.launch(enable_queue=True)
        demo.queue().launch()