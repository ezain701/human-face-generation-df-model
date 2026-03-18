from diffusers import StableDiffusionPipeline
import torch

print("Loading stable diffusion...")
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe = pipe.to("cuda")
print("Pipeline loaded on GPU")

image = pipe("a photo of a beautiful woman's smiling face, portrait, facing forward, no teeth showing").images[0]
image.save("test_face.png")
