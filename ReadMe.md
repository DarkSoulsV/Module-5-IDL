# Monet-Style Image Generation with DCGAN

This project uses a Deep Convolutional GAN (DCGAN) to generate Monet-style images based on the  
Kaggle competition **"I'm Something of a Painter Myself"**.

The model learns from a small dataset of real Monet paintings and tries to produce new images  
that capture similar colors, textures, and artistic patterns.  
The final result is a set of **10,000 generated Monet-style images**, uploaded as a Kaggle submission.

---

## Project Overview

- Trained a **DCGAN** from scratch using TensorFlow/Keras  
- Input: random noise vector  
- Output: fully generated 64×64 Monet-style images  
- Upscaled to 256×256 for Kaggle evaluation  
- Created a submission ZIP (`images.zip`) containing 10k generated images  
- Achieved **Public MiFID Score: 405.56** on Kaggle  

This score places the model around the middle of the leaderboard, which is a solid result for a compact DCGAN.

---

## Model Architecture

### **Generator**
- Dense layer → Reshape  
- Conv2DTranspose blocks for upsampling  
- BatchNorm + ReLU  
- Final `tanh` output layer  

### **Discriminator**
- Strided Conv2D layers for downsampling  
- LeakyReLU activations  
- Optional Dropout  
- Final Dense layer for real/fake prediction  

Both networks are trained adversarially using a custom training loop (`tf.GradientTape`).

---

## Dataset

The training images come from the Kaggle dataset:

**Monet Paintings (256×256 JPGs)**  
https://www.kaggle.com/competitions/gan-getting-started

The project supports:
- **Local dataset loading**
- **Kaggle Notebook execution** (auto-detects `/kaggle/input`)

---

## Training

Training consists of alternating updates:

1. Train discriminator on real + fake images  
2. Train generator to fool the discriminator  
3. Track both losses  
4. Save preview images during training  

The training remained stable (no collapse) and produced consistent samples.

---

## Generating 10k Images

After training, the notebook:

- Generates **10,000 Monet-style** images  
- Upscales them from 64×64 → 256×256  
- Saves each image as a JPG  
- Packs everything into a single ZIP for Kaggle submission  

---

## Kaggle Results

**Public Score: 405.56**

This confirms that the model captured key Monet characteristics, although higher-resolution  
models (CycleGAN, StyleGAN2, etc.) could improve the score further.

---

## Repository Structure

