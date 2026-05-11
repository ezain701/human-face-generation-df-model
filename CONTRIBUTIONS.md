# Contributions

This project was developed collaboratively by all five group members. The overall code design, model structure, experiment planning, and integration decisions were discussed and shaped collectively.

## Role Breakdown

| Pranav Pokhrel | 

- I started working on our base architecture on the 23rd of March alongside Hakan Demirer, where we first focused on understanding the DDPM pipeline and how its components connect-designing and refining the U-Net backbone used for noise prediction, I structured the model around an encoder-decoder design with residual blocks, skip connections, self-attention layers, downsampling, upsampling, and timestep embeddings, with many of these design choices developed through trial and error, including the separation of a modular forward and reverse diffusion to reduce error propagation.

- Through out easter break, starting the 4th of April, I experimented with multiple different set ups with different u-net channel sizes, batch sizes, image sizes, attention placements, and block configurations to improve training stability and generation quality. (however, my logs were overwritten on accident and i wasn't able to reproduce them since i was experimenting with 27000 images (each run lasted 2-3 days on our lab computers), though some of them have been added along with a timestamped screenshot in the branch "older logs")

- Proceeded to help with report writing and visualizations upon completion of experiments.

- On the 7th of may I I built a lightweight transformer-based text encoder to enable prompt-conditioned image generation in the DDPM pipeline. It converts text prompts into fixed-dimensional embeddings that are passed into the U-Net during de-noising, allowing the model to generate images based on textual guidance. The encoder uses a custom deterministic tokenizer, token embeddings, positional embeddings, transformer encoder layers, multi-head self-attention, GELU activations(smoother than relu since it's interpreted stochastically), pooling, and layer normalization. Its tokenizer uses hashed token IDs, so it does not require a saved vocabulary. This made the encoder simpler, more lightweight, and easier to integrate with the project’s training constraints. I also experimented with different embedding sizes, prompt lengths, transformer depths, and conditioning methods, while comparing it against CLIP-based prompting to evaluate prompt alignment and generation quality. |


| Hakan Demirer | 

Worked across the main training and generation pipeline, including data loading, model training workflow, checkpointing, sampling, and general integration, excluding the FID calculation and cosine scheduling components. Also contributed to experimentation, result analysis, and report writing. |


| Zain Ul Abideen | 

- Implemented and extended major components of the DDPM training pipeline, including EMA integration, cosine learning rate scheduling, checkpoint resume logic, experiment management infrastructure, and evaluation workflows.
- Designed, coordinated, and ran the majority of large-scale experiments, including long-duration EMA + cosine LR training runs up to 600 epochs, model-capacity studies, cosine noise schedule experiments, gradient accumulation experiments, and reduced-resolution evaluations.
- Led experiment tracking, checkpoint management, sample generation, FID benchmarking, and qualitative evaluation figure creation, including progression grids across experiments.
Reviewed and validated all major code changes, pull requests, and commits throughout the project to ensure training stability,reproducibility, experiment consistency, and to prevent unstable or incorrect code from being merged into the main branch.
- Helped Matthew implement cosine learning rate scheduling and assisted Hakan in integrating warmup scheduling together
with cosine LR while addressing scheduler interaction, resume-state handling, and optimizer consistency issues.
- Investigated training stability issues related to schedulers, EMA integration, resume logic, and optimizer state restoration across multiple training configurations.
- Managed the GitHub repository workflow, including branch coordination, PR reviews, experiment organization, and final integration.
- Wrote and refined substantial portions of the report, including methodology, experiments, stabilization techniques, results analysis, qualitative evaluation.
- Coordinated final report preparation, formatting, repository release, and coursework submission.  |


| Kevin O'Shaughnessy | 

Once our DDPM pipeline was created, I added unit tests to it to help me understand how the code works at a low level. I started a 100 epoch training run on 29th March. At this stage we were generating 256x256 images and the process took around 24 hours. We subsequently agreed 128x128 would be our default. I recommended the Latex conference template for our report. Then I researched the literature, wrote the literature review, and understood that although we already had a good recreation of the Ho et al. paper there were many subsequent innovations to help us improve our accuracy. I implemented several of these, including cosine noise scheduler and accumulated gradients. Some changes such as Kernel Inception Distance were agreed to be discarded to focus on the core work and reduce the report length, but were useful for learning. Involved in general report editing and reviewed several PRs. |

| Matthew Osborne | 

Worked on cosine scheduling components, including cosine learning-rate scheduling and cosine noise scheduling experiments. Also contributed to the shared code design process, project integration, experimentation, result analysis, and report writing. |
