# Adversarial-Image-Cloaking-Tool

A PyTorch-based adversarial optimization tool that adds imperceptible perturbations to images in order to disrupt deep neural network feature extraction.

This project generates L∞-bounded perturbations that reduce feature similarity between original and modified images under a pretrained ResNet50 model.

---

## Motivation

Modern AI systems rely heavily on feature embeddings extracted from deep convolutional networks.  
This tool explores adversarial techniques to reduce feature similarity while keeping visual changes minimal to the human eye.

The objective is to:
- Preserve perceptual image quality
- Minimize cosine similarity in feature space
- Maintain robustness against common preprocessing transformations

---

Given an input image:

1. Extract feature embeddings using a frozen pretrained ResNet50.
2. Optimize a perturbation `delta` such that:
   - Cosine similarity between original and adversarial features is minimized.
   - Perturbation remains within an L∞ bound.
3. Apply robustness-aware transforms (resize, blur, crop, color jitter) during optimization.
4. Regularize with:
   - Total Variation (TV) loss for smoothness
   - L2 penalty for stability
5. Use this in terminal:
python mls.py --input ./{folder name}/{your input image.png/jpg} --output ./{folder name}/{output name.png/jpg} --eps {higher=more noise} --steps 180 --tv-weight 0.005 --l2-weight 0.001 --max-size 1080 --device cuda(for nvidia gpus)
 
 The final image remains visually similar but differs significantly in feature space.

---

## Tech Stack

- Python
- PyTorch
- Torchvision
- PIL
- NumPy
- Pretrained ResNet50

---

