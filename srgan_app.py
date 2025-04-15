import gradio as gr
import torch
from PIL import Image
import numpy as np
from src.model import Generator  # <---- Import the model

# Load the generator
generator = Generator()
generator.load_state_dict(torch.load("models/srgan/srgenerator.pth", map_location="cpu"))
generator.eval()

# Super-resolution function
def super_resolve(image: Image.Image):
    image = image.convert("RGB")
    image_lr = image.resize((64, 64), resample=Image.BICUBIC)
    img_array = np.array(image_lr).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
    img_tensor = img_tensor * 2 - 1  # Normalize to [-1, 1]

    with torch.no_grad():
        sr_tensor = generator(img_tensor).clamp(-1, 1)

    sr_tensor = (sr_tensor + 1) / 2
    sr_image = sr_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    sr_image = Image.fromarray((sr_image * 255).astype(np.uint8))

    return image, image_lr, sr_image

# Gradio UI
demo = gr.Interface(
    fn=super_resolve,
    inputs=gr.Image(type="pil", label="Upload an image"),
    outputs=[
        gr.Image(type="pil", label="Original"),
        gr.Image(type="pil", label="Low Resolution"),
        gr.Image(type="pil", label="Super Resolution")
    ],
    title="SRGAN Demo",
    description="Upload an image to see SRGAN in action.",
)

if __name__ == "__main__":
    demo.launch(share=True)