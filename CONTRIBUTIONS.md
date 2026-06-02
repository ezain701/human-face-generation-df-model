# Contributions

This project was developed collaboratively by all five group members. The overall code design, model structure, experiment planning, and integration decisions were discussed and shaped collectively.

## Role Breakdown

| Pranav Pokhrel | 

- I started working on our base architecture on the 23rd of March alongside Hakan Demirer, where we first focused on understanding the DDPM pipeline and how its components connect-designing and refining the U-Net backbone used for noise prediction, I structured the model around an encoder-decoder design with residual blocks, skip connections, self-attention layers, downsampling, upsampling, and timestep embeddings, with many of these design choices developed through trial and error, including the separation of a modular forward and reverse diffusion to reduce error propagation.

- Through out easter break, starting the 4th of April, I initiated experimention with multiple different set ups with different u-net channel sizes, batch sizes, image sizes, attention placements, and block configurations to improve training stability and generation quality. (however, my logs were overwritten on accident and i wasn't able to reproduce them since i was experimenting with 27000 images (each run lasted 2-3 days on our lab computers), though some of them have been added along with a timestamped screenshot in the branch "older logs")

- Ran the 27k images experiment on 64x64 images to produce an FID of 16.71
  
- Proceeded to help with report writing and visualizations upon completion of experiments.

- On the 7th of may I I built a lightweight transformer-based text encoder to enable prompt-conditioned image generation in the DDPM pipeline. It converts text prompts into fixed-dimensional embeddings that are passed into the U-Net during de-noising, allowing the model to generate images based on textual guidance. The encoder uses a custom deterministic tokenizer, token embeddings, positional embeddings, transformer encoder layers, multi-head self-attention, GELU activations(smoother than relu since it's interpreted stochastically), pooling, and layer normalization. Its tokenizer uses hashed token IDs, so it does not require a saved vocabulary. This made the encoder simpler, more lightweight, and easier to integrate with the project’s training constraints. I also experimented with different embedding sizes, prompt lengths, transformer depths, and conditioning methods, while comparing it against CLIP-based prompting to evaluate prompt alignment and generation quality. |


| Hakan Demirer | 

- Built the initial diffusion model framework, including the project structure, training pipeline, generation scripts, and core model components.
- Implemented the main training entry point (main.py) with argument parsing, device detection, model initialization, optimizer setup, checkpoint resume logic, and training orchestration.
- Developed the Trainer class (training/trainer.py) with the complete training loop, checkpointing system,  sample generation during training, loss logging, and progress tracking.
- Created the generation script (generate.py) for producing images from trained checkpoints with configurable parameters.
- Implemented data loading infrastructure (data/dataset.py) including CelebA-HQ and Butterfly dataset classes with preprocessing, normalization, and dataloader creation.
- Implemented the noise scheduling module (models/noise_schedule.py) with beta schedule computation and cumulative alpha calculations.
- Set up the project structure with proper Python packaging, dependencies (requirements.txt), and .gitignore configuration.
- Scaled up the dataset handling to support larger training runs (81,000 images) with updated preprocessing including CenterCrop transformation.
- Implemented learning rate warmup scheduling feature with configurable warmup steps parameter.
- Ran the 81K-image warmup experiment achieving FID score of 41.31, demonstrating the effectiveness of warmup on larger datasets.
- Integrated warmup scheduling with the existing scheduler infrastructure, ensuring proper checkpoint state management and resume behavior.
- Addressed code review feedback throughout the project, including scheduler conflict resolution.
- Contributed to experimentation, result analysis, and report writing. | 

| Zain Ul Abideen | 

- Implemented the initial end-to-end DDPM training pipeline and created the foundational project structure, establishing the baseline codebase and integration flow that the later experiments, stabilisation techniques, and architectural improvements were built upon.
- Implemented and extended major components of the DDPM training pipeline, including EMA integration, cosine learning rate scheduling, checkpoint resume logic, experiment management infrastructure, and evaluation workflows.
- Designed, coordinated, and ran the majority of large-scale experiments, including long-duration EMA + cosine LR training runs up to 600 epochs, model-capacity studies, cosine noise schedule experiments, gradient accumulation experiments, and reduced-resolution evaluations.
- Led experiment tracking, checkpoint management, sample generation, FID benchmarking, and qualitative evaluation figure creation, including progression grids across experiments.
- Reviewed and validated all major code changes, pull requests, and commits throughout the project to ensure training stability,reproducibility, experiment consistency, and to prevent unstable or incorrect code from being merged into the main branch.
- Helped Matthew implement cosine learning rate scheduling and assisted Hakan in integrating warmup scheduling together
with cosine LR while addressing scheduler interaction, resume-state handling, and optimizer consistency issues.
- Investigated training stability issues related to schedulers, EMA integration, resume logic, and optimizer state restoration across multiple training configurations.
- Managed the GitHub repository workflow, including branch coordination, PR reviews, experiment organization, and final integration.
- Wrote and refined substantial portions of the report, including methodology, experiments, stabilization techniques, results analysis, qualitative evaluation.
- Coordinated final report preparation, formatting, repository release, and coursework submission.  |


| Kevin O'Shaughnessy | 

* Shortly after our DDPM pipeline was created, I added many unit tests to validate component behaviour and better understand the code at a low level. 
* Initiated 100 epoch training run on 29th March. At this stage we were generating 256x256 images and the process took around 24 hours. We subsequently agreed 128x128 would be our default. 
* Recommended the Latex conference template for our report and created the outline with main sections are pointers for teammates. 
* Deeply researched the literature including GAN, FID, and DDPM papers, wrote the literature review, understanding that although we already had a good recreation of the Ho et al. paper there were many subsequent innovations to help us improve our accuracy.
* Identified and implemented several post-DDPM improvements not present in the initial pipeline: cosine noise scheduling, gradient accumulation for larger effective batch sizes, DDIM sampling for accelerated image generation, and Kernel Inception Distance (KID) as a complement to FID. Some features, at the time of writing, are not merged due to a desire to focus on the core work and reduce the report length, but were used sucessfully in some training runs.
* Investigated training instabilities including degenerate sample generation (near-black and near-white outputs), tracing the root cause to AdaGN scale parameter drift and identifying initialisation corrections.
* Reviewed pull requests and contributed to report editing and revision.

| Matthew Osborne | 

- Developed and implented the cosine scheduling aspects of the model this includes; the development of cosine learning-rate scheduling and cosine noise scheduling experiments.
- In addition to the aforementioned, I supported my above collaborators in the shared code design, project integration, experiments and associated results and analysis.
- Supported collation of the report. |
