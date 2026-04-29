# Abstract Generative modelling aims to learn complex data distributions in order to generate realistic samples. In this project, we implement a diffusion-based model for unconditional human face generation using the CelebA-HQ dataset. The model is based on the Denoising Diffusion Probabilistic Model framework, where a neural network is trained to reverse a gradual Gaussian noising process. Experiments were conducted using a restricted dataset of 2700 training images and 300 test images, with performance evaluated using the Fréchet Inception Distance. We systematically investigated the effects of image resolution, model capacity, training duration, learning rate, and stabilisation techniques. Results show that longer training substantially improves generation quality, but also introduces instability when no stabilisation is used. Exponential Moving Average improves robustness and sample quality, reducing FID from approximately 234 to 173 in an early controlled comparison, and enabling continued improvement over longer training. The best results were obtained by combining EMA with cosine learning rate scheduling, achieving a final FID of 41.82 after 600 epochs. Qualitative results also showed clearer facial structure, improved symmetry, and reduced artifacts. Overall, the project demonstrates that diffusion models can generate realistic human faces even under limited data and computational constraints, with training stability playing a key role in final performance.

# Introduction Generative modelling aims to learn data distributions in order to generate realistic samples. In image generation, this involves producing images that are visually indistinguishable from real data, with applications in content creation and data augmentation.

Human face generation is a challenging task due to complex structures and fine details, making it a strong benchmark for generative models. While Generative Adversarial Networks (GANs) \[4\] have achieved strong results, they often suffer from instability and mode collapse.

Diffusion models provide a more stable alternative by learning to iteratively transform noise into structured images through a denoising process. This approach has achieved state-of-the-art performance across multiple domains \[8\].

In this work, we implement a diffusion-based model for human face generation using the CelebA-HQ dataset. We investigate the effects of architecture, resolution, training duration, and stabilization techniques such as Exponential Moving Average (EMA). Performance is evaluated using the Fréchet Inception Distance (FID), with significant improvements achieved through systematic experimentation.

Literature Review  
Generative modeling aims to learn a probability distribution over high-dimensional data such that new samples can be drawn from it. Image synthesis has long served as a proving ground for generative methods, and for much of the last decade the dominant approach was Generative Adversarial Networks (GANs) \\cite{b6}. GANs frame image generation as a two-player game between a generator that produces candidate samples from random noise and a discriminator that distinguishes them from real data.

Despite producing visually impressive results, GANs are notoriously difficult to train: the adversarial objective is prone to instability, mode collapse reduces the diversity of generated samples, and convergence is sensitive to architectural and hyperparameter choices. These limitations motivated the search for generative models with more stable training dynamics and stronger likelihood-based foundations.

Diffusion models provide such an alternative. The core idea, first proposed by Sohl-Dickstein et al. \[4\] and drawn from nonequilibrium thermodynamics, is to define a forward process that gradually corrupts data by adding Gaussian noise over many timesteps, and then to learn a reverse process that removes noise step by step, recovering samples from the data distribution.

Ho et al. \[3\] made this framework practical with Denoising Diffusion Probabilistic Models (DDPMs), and this formulation of DDPM forms the basis of the model implemented in this project. Song et al. \[2\] unified discrete-time denoising diffusion with continuous-time score-based generative modeling under a single stochastic differential equation framework, showing that both approaches can be viewed as learning the score function of a data distribution perturbed by noise at varying scales. 

Shortly afterwards, Dhariwal and Nichol \[1\] demonstrated empirically that diffusion models outperform GANs on standard image synthesis benchmarks across multiple resolutions, through a combination of architectural improvements, a learned reverse-process variance, and classifier-based guidance for conditional generation.  
       
Subsequent work has identified several refinements to the original DDPM formulation that improve sample quality at modest implementation cost. Nichol and Dhariwal \[8\] introduced a cosine noise schedule in which the cumulative signal-preservation factor decays as a shifted squared cosine. The cosine schedule destroys image information more gradually across timesteps, avoiding the rapid saturation observed at late timesteps under linear scheduling, and yields improved FID at equivalent training cost. 

Dhariwal and Nichol \[1\] further showed that adaptive group normalization — in which the affine parameters of group normalization are predicted from the timestep embedding rather than learned as constants — provides a stronger form of timestep conditioning than the additive injection used in the original DDPM. The use of exponential moving averages of model parameters during sampling, while not introduced by any single paper, has become standard practice across the diffusion literature for stabilizing sample quality. 

Progress in generative image modeling depends on standardized benchmarks and quantitative evaluation. For unconditional face generation, the CelebA-HQ dataset introduced by Karras et al. \[7\] has become a widely adopted benchmark, providing 30,000 high-quality face images and supporting evaluation across GANs, score-based approaches, and diffusion models. 

To assess sample quality, the Fréchet Inception Distance (FID) proposed by Heusel et al. \[5\] has emerged as the standard metric. FID embeds both real and generated images into the feature space of a pretrained Inception-v3 network, fits a multivariate Gaussian to each set of features, and computes the Fréchet distance between the two Gaussians. 

Methodology

**Diffusion Model Framework**  
This work is based on the Denoising Diffusion Probabilistic Model (DDPM) framework. Diffusion models consist of a forward process and a reverse process. The forward process gradually adds Gaussian noise to an image over a fixed number of timesteps until it becomes pure noise. The reverse process is learned by a neural network, which iteratively removes noise to reconstruct the original image.

The model is trained to predict the noise added at each timestep. Starting from random noise, the learned reverse process enables generation of realistic images through successive denoising steps.

**Dataset and Preprocessing**  
Experiments were conducted on the CelebA-HQ dataset, restricted to 2700 training images and 300 test images as specified.

Images were:

* resized to the required resolution (64×64 or 128×128)  
* normalized for stable training

No additional augmentation was applied to ensure consistency across experiments.

**Training Setup**  
Models were trained using the Adam optimizer with a learning rate of 0.0002 and a batch size of 1\. The diffusion process used 1000 timesteps.

Key variables explored include:

* image resolution  
* model capacity (base channels)  
* number of training epochs

## **Stabilization Techniques**

To improve training stability, Exponential Moving Average (EMA) was applied to model parameters, maintaining a smoothed version of the weights during training. EMA reduces noisy updates and improves sample quality during generation.

Additionally, gradient clipping was used to limit large gradient updates and further stabilize training.

**Evaluation Metric**  
Performance was evaluated using the Fréchet Inception Distance (FID), which measures the similarity between real and generated image distributions. Lower FID indicates better sample quality and diversity.

## **Experiments** All experiments were conducted using a Denoising Diffusion Probabilistic Model (DDPM) trained on the CelebA-HQ dataset. In accordance with the coursework specification, the dataset was limited to 2700 training images and 300 test images. Model performance was evaluated using the Fréchet Inception Distance (FID), computed between 300 generated images and the test set.

The baseline configuration was defined as:

Image resolution: **128 × 128**  
Base channels: **64**  
Batch size: **1**  
Learning rate: **0.0002**  
Diffusion timesteps: **1000**

Experiments were designed to systematically explore the impact of:

1\. Model capacity  
2\. Image resolution  
3\. Training duration  
4\. Learning rate  
5\. Training stabilization techniques

Baseline Performance

The baseline experiment (Experiment B) established a reference point using the default configuration. While it produced visually plausible face structures, the FID score remained relatively high, indicating limited sample quality and diversity.

This baseline provided a foundation for subsequent improvements through architectural and training modifications

Effect of Model Capacity  
To evaluate the effect of model capacity, experiments were conducted with varying base channel sizes.

At early training (20 epochs), both reducing and increasing model capacity relative to the baseline led to improved FID scores. Specifically, the smaller model (Experiment C, ch32) achieved a lower FID than the baseline, and the larger model (Experiment G, ch96) also improved upon the baseline. This suggests that model capacity alone does not directly determine early performance, and that optimization dynamics play a significant role during initial training.

However, subsequent experiments focused on extending training primarily with the baseline configuration (ch64), where consistent and significant improvements were observed as training duration increased. This configuration ultimately led to the best-performing models in later stages.

These results indicate that while alternative model sizes may perform well in early training, the moderate configuration (ch64) provides a reliable and stable foundation for extended training and further optimization.

Effect of Image Resolution  
The effect of image resolution was investigated by comparing 128×128 and 64×64 configurations.

Lower resolution models (Experiment D) achieved a significantly lower FID score compared to the baseline. This improvement is expected, as generating lower-resolution images is an easier task, resulting in better quantitative performance. However, visual inspection revealed reduced detail and realism in generated faces. While lower resolution simplifies the learning problem, it limits the model’s ability to generate high-quality images.

Overall, 128×128 resolution provided better qualitative results and improved scalability, and was therefore retained for subsequent experiments.

Effect of Training Duration

Training duration was found to have a significant impact on model performance.

For the baseline configuration without additional stabilisation techniques, increasing the number of epochs initially led to improvements in FID:

* 100 epochs → FID ≈ 181  
* 150 epochs → FID ≈ 122  
* 200 epochs → FID ≈ 118

However, extending training further without stabilisation resulted in degradation:

* 250 epochs → FID ≈ 159

This indicates that, in the absence of stabilisation, prolonged training can lead to instability and reduced sample quality.

This indicates that, in the absence of stabilisation, prolonged training can lead to instability and reduced sample quality.

To address this, Exponential Moving Average (EMA) was introduced, which enabled stable training over longer durations. With EMA, performance improved consistently with increased training:

* 200 epochs → FID ≈ 95  
* 250 epochs → FID ≈ 76  
* 300 epochs → FID ≈ 62

Further improvements were achieved by combining EMA with a cosine learning rate scheduler. Extending training up to 600 epochs resulted in the best performance:

* 350 epochs → FID ≈ 46.7  
* 450 epochs → FID ≈ 42.6  
* 600 epochs → FID ≈ 41.8

While overall performance improved with longer training, results beyond 450 epochs exhibited some fluctuation, suggesting diminishing returns and minor instability in later stages.

Effect of Learning Rate  
A lower learning rate (Experiment J, **1e-4**) was also evaluated. While this configuration resulted in more gradual training, it led to slower convergence and inferior FID scores compared to the baseline learning rate of 0.0002.

Therefore, 0.0002 was retained as the optimal learning rate, providing a better balance between convergence speed and stability.

EMA for Training Stabilization  
To address instability in longer training runs, Exponential Moving Average (EMA) was introduced. EMA maintains a smoothed version of model parameters, reducing the impact of noisy updates during optimization.

EMA was applied to the best-performing baseline configuration, which was used consistently across the F-series experiments.

A controlled experiment was conducted to validate its effectiveness over 10 epochs:

* Raw model → FID ≈ 234  
* EMA model → FID ≈ 173

This demonstrated a substantial improvement even in early training.

EMA was then applied to longer training runs, producing significant and consistent gains:

* 200 epochs (EMA) → FID ≈ 95  
* 250 epochs (EMA) → FID ≈ 76  
* 300 epochs (EMA) → FID ≈ 62

However, extending training further with EMA alone did not continue to improve performance:

* 350 epochs (EMA) → FID ≈ 63.21

This indicates that while EMA stabilises training and prevents degradation, it does not fully address diminishing returns at later stages.

Effect of Cosine Learning Rate Scheduling

To further improve training stability and convergence in later stages, a cosine learning rate scheduler was introduced in combination with EMA.

This modification enabled effective scaling to longer training durations and resulted in substantial improvements over EMA alone:

* 350 epochs → FID ≈ 46.7  
* 450 epochs → FID ≈ 42.6  
* 600 epochs → FID ≈ 41.8

Compared to EMA-only training, the addition of cosine learning rate scheduling significantly improved performance, particularly in the later stages of training where improvements had previously plateaued.

However, results beyond 450 epochs exhibited some fluctuations (e.g., increase at 500 epochs), indicating minor instability and diminishing returns despite the overall downward trend.

## Qualitative Results Generated samples show a clear progression in visual quality as training duration increases. Early-stage models (e.g., 20–100 epochs) produce noisy and poorly structured faces, while mid-stage models (150–300 epochs) begin to capture more coherent facial structures.

## With extended training and the use of EMA, later models produce significantly improved results, exhibiting better symmetry, more consistent facial features, and reduced visual artifacts.

## EMA-based models consistently generate smoother outputs compared to their non-EMA counterparts, which often display instability and distortions in later training stages.

## Further improvements were observed when combining EMA with cosine learning rate scheduling, with late-stage models (350–600 epochs) producing sharper and more visually coherent faces.

## While the generated images do not fully match the fidelity of real samples, the final model captures key characteristics of the data distribution, including facial structure and general appearance.

## Results Summary

Table 1 summarizes the most important experimental results.

| Experiment | Configuration | FID |
| :---- | :---- | :---- |
| B | 128px, ch64, batch 1, 20 epochs | 309.14 |
| C | 128px, ch32, batch 1, 20 epochs | 261.26 |
| D | 64px, ch64, batch 1, 20 epochs | 216.12 |
| E | 128px, ch64, batch 1, 50 epochs | 282.86 |
| F100 | 128px, ch64, batch 1, 100 epochs | 181.60 |
| G | 128px, ch96, batch 1, 20 epochs | 247.07 |
| H | 128px, ch32, batch 1, 50 epochs | 209.64 |
| I | 128px, ch96, batch 1, 50 epochs | 257.59 |
| K | 128px, ch96, lr=1e-4, 20 epochs | 247.59 |
| J100 | 128px, ch64, lr=1e-4, 100 epochs | 240.12 |
| J250 | 128px, ch64, lr=1e-4, 250 epochs | 159.86 |
| F150 | 128px, ch64, 150 epochs | 122.23 |
| F200 | 128px, ch64, 200 epochs | 118.30 |
| F250 raw | 128px, ch64, 250 epochs | 159.40 |
| EMA raw check | ch64, 10 epochs (no EMA) | \~234 |
| EMA check | ch64, 10 epochs (EMA) | \~173 |
| F200 EMA | 128px, ch64, EMA, 200 epochs | \~95 |
| F250 EMA | 128px, ch64, EMA, 250 epochs | \~76 |
| F300 EMA | 128px, ch64, EMA, 300 epochs | \~62 |
| F350 EMA | 128px, ch64, EMA, 350 epochs | \~63.21 |
| F350 EMA \+ cosine LR | 128px, ch64, EMA, cosine LR, 350 epochs | 45.02 / |
| F400 EMA \+ cosine LR | 128px, ch64, EMA, cosine LR, 400 epochs | 44.65 |
| F450 EMA \+ cosine LR | 128px, ch64, EMA, cosine LR, 450 epochs | 42.64 |
| F500 EMA \+ cosine LR | 128px, ch64, EMA, cosine LR, 500 epochs | 45.92 |
| F550 EMA \+ cosine LR | 128px, ch64, EMA, cosine LR, 550 epochs | 43.90 |
| F600 EMA \+ cosine LR | 128px, ch64, EMA, cosine LR, 600 epochs | 41.82 |
| F300 EMA \+ cosine noise | 128px, ch64, EMA, cosine noise | 314 |
| F100 81K Images | 128px, ch128, batch 16, 100 epochs, warm-up steps | 41.31 |
|  |  |  |
|  |  |  |

## Conclusion

The project implemented and evaluated a diffusion-based approach for human face generation using the CelebA-HQ dataset. The model was trained under the DDPM framework, where image generation is achieved by learning to iteratively denoise random Gaussian noise into realistic face images. Across the experiments, performance was evaluated using FID and supported by qualitative inspection of generated samples.

The results show that training duration has a major impact on sample quality. The baseline model improved substantially as training increased from 100 to 200 epochs, with FID decreasing from approximately 181 to 118\. However, extending training to 250 epochs without additional stabilisation caused performance to degrade, increasing FID to around 159\. This suggests that longer training alone is not sufficient, as the model can become unstable without techniques that smooth or regulate optimisation.

Exponential Moving Average was found to be an effective stabilisation method. EMA consistently improved sample quality compared to the raw model and allowed longer training runs to remain stable. With EMA, FID improved to approximately 95 at 200 epochs, 76 at 250 epochs, and 62 at 300 epochs. The strongest result under the restricted 2700-image training setup was achieved when EMA was combined with cosine learning rate scheduling. This configuration achieved an FID of 41.82 at 600 epochs, with visually clearer and more coherent generated faces.

A further large-scale experiment showed the importance of dataset size, batch size, and model capacity. Training on 81K images using 128×128 resolution, 128 base channels, batch size 16, and 100 epochs achieved an FID of 41.31. This was the best overall result and slightly outperformed the 600-epoch restricted-dataset model. This suggests that access to more training data and larger batch sizes can improve convergence efficiency.

Other experimental factors also influenced performance. Lower resolution images produced better FID scores in early experiments, but the generated samples lacked fine detail and realism. Therefore, 128×128 resolution was retained as a better trade-off between quantitative performance and visual quality. Changes in model capacity showed that both smaller and larger models could improve early results, but the 64-channel configuration provided the most reliable foundation for extended training under the restricted dataset setting. A lower learning rate led to slower convergence and inferior results compared to the baseline learning rate of 0.0002.

Despite these improvements, several limitations remain. Most experiments were conducted using only 2700 training images and a batch size of 1, which limited diversity, slowed training, and made optimisation noisier. Although the final models captured general facial structure, symmetry, and appearance, the generated images still do not fully match the fidelity of real samples. In addition, improvements beyond 450 epochs showed diminishing returns and some fluctuation, indicating that later-stage training remains difficult to stabilise.

Future work could improve the model by training consistently on larger datasets, using larger batch sizes, and exploring more advanced diffusion architectures. Additional improvements could include learned variance, classifier-free guidance, or higher-resolution training. More efficient sampling methods could also be investigated to reduce generation time. Overall, the experiments demonstrate that diffusion models are effective for human face generation, and that both stabilisation techniques and training scale are essential for achieving strong performance.

## References (IEEE format)

**\[1\]** P. Dhariwal and A. Nichol, "Diffusion models beat GANs on image synthesis," in *Advances in Neural Information Processing Systems*, vol. 34, pp. 8780–8794, 2021\.

**\[2\]** Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon, and B. Poole, "Score-based generative modeling through stochastic differential equations," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2021\.

**\[3\]** M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, and S. Hochreiter, "GANs trained by a two time-scale update rule converge to a local Nash equilibrium," in *Advances in Neural Information Processing Systems*, vol. 30, pp. 6626–6637, 2017\.

**\[4\]** I. J. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio, "Generative adversarial nets," in *Advances in Neural Information Processing Systems*, vol. 27, pp. 2672–2680, 2014\.

**\[5\]** T. Karras, S. Laine, and T. Aila, "A style-based generator architecture for generative adversarial networks," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, pp. 4401–4410, 2019\.

**\[6\]** T. Kynkäänniemi, T. Karras, S. Laine, J. Lehtinen, and T. Aila, "Improved precision and recall metric for assessing generative models," in *Advances in Neural Information Processing Systems*, vol. 32, 2019\.

**\[7\]** T. Salimans, I. Goodfellow, W. Zaremba, V. Cheung, A. Radford, and X. Chen, "Improved techniques for training GANs," in *Advances in Neural Information Processing Systems*, vol. 29, 2016\.

**\[8\]** L. Yang, Z. Zhang, Y. Song, S. Hong, R. Xu, Y. Zhao, W. Zhang, B. Cui, and M.-H. Yang, "Diffusion models: A comprehensive survey of methods and applications," *ACM Computing Surveys*, vol. 56, no. 4, pp. 1–39, 2024\.

**\[9\]** O. Ronneberger, P. Fischer, and T. Brox, "U-Net: Convolutional networks for biomedical image segmentation," in *Proc. Int. Conf. Med. Image Comput. Comput.-Assisted Intervention (MICCAI)*, pp. 234–241, 2015\.

## Reference Implementations

## 

## Original DDPM implementation [https://github.com/hojonathanho/diffusion](https://github.com/hojonathanho/diffusion)

## Denoising Diffusion Probabilistic Model in Pytorch

[https://github.com/lucidrains/denoising-diffusion-pytorch/](https://github.com/lucidrains/denoising-diffusion-pytorch/) 

## Improved Diffusion [https://github.com/openai/improved-diffusion](https://github.com/openai/improved-diffusion)

Guided Diffusion  
[https://github.com/openai/guided-diffusion](https://github.com/openai/guided-diffusion)

## Links and commentary on Papers \[not for final report\]

Original Goodfellow GANs paper 2014 [https://arxiv.org/pdf/1406.2661](https://arxiv.org/pdf/1406.2661)

Diffusion Models beat GANs [https://arxiv.org/pdf/2105.05233](https://arxiv.org/pdf/2105.05233)

StyleGAN [https://arxiv.org/abs/1812.04948](https://arxiv.org/abs/1812.04948)

FID [https://papers.nips.cc/paper\_files/paper/2017/file/8a1d694707eb0fefe65871369074926d-Paper.pdf](https://papers.nips.cc/paper_files/paper/2017/file/8a1d694707eb0fefe65871369074926d-Paper.pdf)

Score based Generative Modeling through Stochastic Differential Equations  
[https://arxiv.org/abs/2011.13456](https://arxiv.org/abs/2011.13456)

Improved Techniques for training GANs [https://arxiv.org/abs/1606.03498](https://arxiv.org/abs/1606.03498) 

# Diffusion Models: A Comprehensive Survey of Methods and Applications

[https://arxiv.org/abs/2209.00796](https://arxiv.org/abs/2209.00796)  
