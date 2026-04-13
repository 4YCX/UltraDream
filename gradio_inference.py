try:
    import gradio as gr
except ImportError:
    print("Please install gradio: `pip install gradio`")
    exit(1)

from pathlib import Path
from typing import Dict, List
from PIL import Image as PILImage
import numpy as np  # 确保numpy先导入，避免初始化问题

# 加载模型相关
from dalle2_laion import ModelLoadConfig, DalleModelManager, utils
from dalle2_laion.scripts import BasicInference, ImageVariation, BasicInpainting

# 配置路径（确保你的config.json在正确位置）
config_path = Path(__file__).parent / 'configs/gradio.example.json'
model_config = ModelLoadConfig.from_json_path(config_path)
model_manager = DalleModelManager(model_config)

output_path = Path(__file__).parent / 'output/gradio'
output_path.mkdir(parents=True, exist_ok=True)


# 生成条件尺度滑块（避免重复渲染）
def create_cond_sliders():
    sliders = [gr.Slider(minimum=0.5, maximum=5, step=0.05, label="Prior Cond Scale", value=1)]
    for i in range(model_manager.model_config.decoder.final_unet_number):
        sliders.append(gr.Slider(minimum=0.5, maximum=5, step=0.05,
                                 label=f"Decoder {i + 1} Cond Scale", value=1))
    return sliders


# Dream功能
def dream(text: str, samples_per_prompt: int, prior_cond_scale: float, *decoder_cond_scales):
    prompts = text.strip().split('\n')[:8]
    prompts = [p for p in prompts if p]
    if not prompts:
        return []
    script = BasicInference(model_manager, verbose=True)
    output = script.run(prompts, prior_sample_count=samples_per_prompt,
                        decoder_batch_size=40, prior_cond_scale=prior_cond_scale,
                        decoder_cond_scale=decoder_cond_scales)
    all_outputs = []
    for _, emb_outputs in output.items():
        for _, imgs in emb_outputs.items():
            all_outputs.extend(imgs)
    return all_outputs


# Variation功能（修复图片上传回调）
def variation(image: PILImage.Image, text: str, num_generations: int, *decoder_cond_scales):
    if image is None:
        return []
    print("Variation prompt:", text)
    img = utils.center_crop_to_square(image.convert('RGB'))
    script = ImageVariation(model_manager, verbose=True)
    output = script.run([img], [text], sample_count=num_generations, cond_scale=decoder_cond_scales)
    all_outputs = []
    for _, imgs in output.items():
        all_outputs.extend(imgs)
    return all_outputs


# Inpaint功能
def inpaint(image_dict: Dict[str, PILImage.Image], text: str, num_generations: int,
            prior_cond_scale: float, *decoder_cond_scales):
    if not image_dict or 'image' not in image_dict or 'mask' not in image_dict:
        return []
    img, mask = image_dict['image'], image_dict['mask']
    img = img.convert('RGB')
    img = utils.center_crop_to_square(img)
    mask = utils.center_crop_to_square(mask)
    script = BasicInpainting(model_manager, verbose=True)
    mask = ~utils.get_mask_from_image(mask)
    output = script.run(images=[img], masks=[mask], text=[text],
                        sample_count=num_generations, prior_cond_scale=prior_cond_scale,
                        decoder_cond_scale=decoder_cond_scales)
    all_outputs = []
    for _, imgs in output.items():
        all_outputs.extend(imgs)
    return all_outputs


# 构建界面（用Blocks+Tab，彻底修复重复渲染与上传无响应）
with gr.Blocks(title="DALL-E 2 Laion") as demo:
    gr.Markdown("# DALL-E 2 Laion Inference")

    with gr.Tab("Dream"):
        with gr.Row():
            text_input = gr.Textbox(placeholder="A corgi wearing a top hat...", lines=8, label="Prompts")
            samples_slider = gr.Slider(minimum=1, maximum=4, step=1, label="Samples per prompt", value=1)
        cond_sliders = create_cond_sliders()
        dream_btn = gr.Button("Generate Dream", variant="primary")
        dream_gallery = gr.Gallery(label="Outputs", columns=2)
        dream_btn.click(dream, inputs=[text_input, samples_slider, *cond_sliders], outputs=dream_gallery)

    with gr.Tab("Variation"):
        with gr.Row():
            img_input = gr.Image(type="pil", source="upload", interactive=True, label="Input Image")
            var_text = gr.Textbox(label="Guiding Prompt")
            var_num = gr.Slider(minimum=1, maximum=6, step=1, label="Number to generate", value=2)
        var_btn = gr.Button("Generate Variation", variant="primary")
        var_gallery = gr.Gallery(label="Outputs", columns=2)
        var_btn.click(variation, inputs=[img_input, var_text, var_num, *cond_sliders[1:]], outputs=var_gallery)

    with gr.Tab("Inpainting"):
        with gr.Row():
            inpaint_img = gr.Image(type="pil", source="upload", tool="sketch", interactive=True, label="Image + Mask")
            inpaint_text = gr.Textbox(label="Prompt for masked area")
            inpaint_num = gr.Slider(minimum=1, maximum=6, step=1, label="Number to generate", value=2)
        inpaint_btn = gr.Button("Generate Inpainting", variant="primary")
        inpaint_gallery = gr.Gallery(label="Outputs", columns=2)
        inpaint_btn.click(inpaint, inputs=[inpaint_img, inpaint_text, inpaint_num, *cond_sliders],
                          outputs=inpaint_gallery)

# 启动（修复本地访问+队列，兼容Gradio3.x）
if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        enable_queue=True,  # 必须启用队列，否则上传/生成无响应
        share=False,
        debug=True  # 开启debug，方便看后端报错
    )