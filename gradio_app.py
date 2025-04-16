import gradio as gr
import torch
from PIL import Image
import numpy as np
from src.model import Generator  # Asegúrate de tener la clase Generator en src/model.py

# Load pretrained SRGAN Generator
generator = Generator()
generator.load_state_dict(torch.load("models/srgan/srgenerator.pth", map_location="cpu"))
generator.eval()

# Super-resolution function
def super_resolve(image: Image.Image):
    display_size = (256, 256)  # For uniform display

    image = image.convert("RGB")
    image_lr = image.resize((64, 64), resample=Image.BICUBIC)

    # Preprocessing
    img_array = np.array(image_lr).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
    img_tensor = img_tensor * 2 - 1  # Normalize to [-1, 1]

    # Inference
    with torch.no_grad():
        sr_tensor = generator(img_tensor).clamp(-1, 1)

    # Postprocessing
    sr_tensor = (sr_tensor + 1) / 2
    sr_image = sr_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    sr_image = Image.fromarray((sr_image * 255).astype(np.uint8))

    return (
        image.resize(display_size),
        image_lr.resize(display_size),
        sr_image.resize(display_size)
    )

# Gradio UI layout
with gr.Blocks(title="SRGAN") as demo:
    gr.Markdown("## Super-Resolution GAN on CelebA Dataset")
    gr.Markdown("Upload an image and watch it upscale from 64×64 to 128×128 using SRGAN.")

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Upload an image")
            run_button = gr.Button("Run SRGAN")

        with gr.Column():
            output_original = gr.Image(type="pil", label="Original Image", width=256, height=256)
            output_lr = gr.Image(type="pil", label="Low Resolution (64×64)", width=256, height=256)
            output_sr = gr.Image(type="pil", label="Super Resolution", width=256, height=256)

    # Define callback
    run_button.click(
        fn=super_resolve,
        inputs=input_image,
        outputs=[output_original, output_lr, output_sr]
    )

# Launch the app
if __name__ == "__main__":
    demo.launch(share=True)

### python srgan_app.py