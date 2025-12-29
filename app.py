"""
多模态检索系统主入口
整合文搜图和图搜图两个功能
"""
import gradio as gr
from text2image import text2image_gr
from image2image import image2image_gr

if __name__ == "__main__":
    gr.close_all()
    with gr.TabbedInterface(
            [text2image_gr(), image2image_gr()],
            ["📝 文到图搜索", "🖼️ 图到图搜索"],
    ) as demo:
        demo.queue().launch(
            server_name="127.0.0.1",
            share=False
        )