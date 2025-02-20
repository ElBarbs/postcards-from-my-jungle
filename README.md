# CART498 (GenAI)

## Assignment 5 - Postcards from My Jungle

### Training the Diffusion Model

To ensure quality, I selected a dataset with high download numbers on Hugging Face. I chose `mertcobanov/animals`, which contains 5,400 color images across 81 animal classes. Due to its large size, I worked with a subset of 1,024 images.

I experimented with different resolutions to optimize image clarity. At 64x64 pixels, the generated outputs were mostly noise. Increasing to 128x128 pixels produced vague shapes, but the colors appeared faded. At 256x256 pixels, the images started to resemble the shape of animals, though they remained somewhat blurry. This resolution provided the best balance between recognizability and training feasibility.

The rest of the preprocessing pipeline remained the same: images were randomly flipped horizontally, converted to tensors, and normalized with a mean of 0.5 and a standard deviation of 0.5.

To refine the model’s output, I increased the number of time-steps in the DDPMScheduler to 1,500 while keeping the optimizer's learning rate at 2e-5. The batch size was set to 16 to balance memory usage, given the higher resolution images.

I trained the model for 100 epochs, and the training took about 3h30.

### Generating Sounds

For sound generation, I used the pre-trained diffusion model `stabilityai/stable-audio-open-1.0`. Many popular Text-to-Audio models on Hugging Face were not publicly available, making this one the most accessible option.

To create sound prompts, I provided `GPT-4o` with a generated image and asked it to describe the fictional animal’s sound. Additionally, I requested a negative prompt to improve output quality:
```
"low bitrate, compressed, muffled, distorted, clipping, static, glitchy"
```

I increased the number of inference steps to 400 while keeping the number of waveforms per prompt at 3.

### Generating Phrases

For text generation, I used the pre-trained transformer model `GPT-4o`. I prompted it to create phrases combining various animal sounds in a way that resembled speech but remained unrecognizable as any known language.

### Possible Improvements

- **Higher Resolution Images**: Using higher-resolution images could enhance output quality, though it would increase computational costs and training time.

- **Modifying Batch Size**: A larger batch size (e.g., 32) can stabilize training, while a smaller one (e.g., 8) may help escape local minima.

- **Exploring Optimizers**: Trying alternatives like RMSprop or adjusting Adam’s parameters may enhance training stability.

- **Switching Sampling Methods**: Opting for an alternative sampler, like "dpmpp-3m-sde," could enhance the quality and detail of the output.
