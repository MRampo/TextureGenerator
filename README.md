# Texture Synthesis with Generative Adversarial Networks

## Overview
This project explores a modern approach to texture synthesis using Generative Adversarial Networks (GANs). The main objective is to generate larger, visually appealing textures from smaller samples by capturing and learning the essential characteristics of the input texture. The methodology includes designing and training GANs that incorporate texture analysis and synthesis, emphasizing feature extraction, pattern recognition, and image generation processes.

## Methodology
The methodology combines various techniques refined over the years:
Architecture: 
-- The generator is an encoder-decoder architecture based on Johnson et al. (2016) with residual blocks to capture large-scale non-stationary behavior
-- The discriminator is a fully convolutional network based on Isola et al. (2016).

## Training
The training procedure involves creating random crops of the original texture and using them as inputs for the generator. The loss function combines adversarial loss (Goodfellow et al., 2014) and style loss (Gatys et al., 2015) computed from a pre-trained VGG-19 model.

## Some Results
The results from our texture synthesis model are quite satisfactory. The generated textures exhibit a high degree of closeness to the target textures. However, it is important to note that some fine-grained details are still missing, which could enhance the overall quality of the textures. This is probably due to the lack of a higher iteration training.

Despite these minor shortcomings, based on the main paper used for this project, by training the model for additional iterations, we could probably achieve even better results. More iterations would allow the model to capture the intricate details that are currently lacking, leading to more refined and accurate textures.

![grass_modello](https://github.com/MRampo/TextureGenerator/assets/37956397/2628ec08-cbb6-4c3f-bd20-b71215f71dd5)
![Water](https://github.com/MRampo/TextureGenerator/assets/37956397/b999d208-e5fc-45b4-a8c1-3ef0eb5f8b3e)
![Wood2](https://github.com/MRampo/TextureGenerator/assets/37956397/0b343c27-81e6-4be2-9d26-5bed58bcfa20)
![dirt](https://github.com/MRampo/TextureGenerator/assets/37956397/07952aa7-6d26-44c5-91bb-56a6057be61f)
