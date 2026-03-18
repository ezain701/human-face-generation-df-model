from diffusers import StableDiffusionXLPipeline
import torch

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", 
    torch_dtype=torch.float16 
)
print("Pipeline loaded on GPU")

# for the full list of parameters see https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl#diffusers.StableDiffusionXLPipeline.__call__
image = pipe("a photo of a beautiful latina woman's face, portrait, facing forward, no teeth showing",
            num_inference_steps=30,      # more steps = higher quality, slower
            guidance_scale=20,          # how closely to follow the prompt (higher = more literal)
            negative_prompt="blurry, distorted, low quality",  # what to avoid
            num_images_per_prompt = 1,
            seed=1                      # for reproducibility
            ).images[0]
image.save("test_face.png")

print("Image saved")
image
