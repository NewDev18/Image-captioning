# Image Captioning with PyTorch: CNN Encoder + LSTM Decoder

## Overview
This repository implements an image captioning model using a CNN encoder and LSTM decoder architecture in PyTorch. The model automatically generates textual descriptions for images by combining computer vision and natural language processing techniques.

## Architecture

### CNN Encoder
The CNN encoder extracts visual features from images:
- Uses a pre-trained CNN (ResNet, VGG, etc.) with the classification layer removed
- Transforms images into feature vectors that capture visual content
- Feature dimension is reduced to match the input requirements of the LSTM

### LSTM Decoder
The LSTM decoder generates captions based on image features:
- Takes the encoded image features as initial input
- Generates words sequentially, with each word prediction depending on previous words
- Uses word embeddings to represent words as dense vectors
- Includes attention mechanism (optional) to focus on different parts of the image

## Training Process

1. **Data Preparation**:
   - Process image dataset (e.g., COCO, Flickr8k(used in this repo))
   - Tokenize captions and build vocabulary
   - Create image-caption pairs for training

2. **Training Loop**:
   - Encoder extracts image features
   - Features are fed to the decoder as initial state
   - Teacher forcing is used during training (use ground truth words as input)
   - Calculate loss using cross-entropy between predicted and actual words
   - Backpropagate and update model parameters

## Inference
During inference:
1. Image is processed by the CNN encoder
2. LSTM decoder generates caption word-by-word
3. Each word is conditioned on previously generated words
4. Generation stops when <END> token is produced or max length is reached

## Requirements
- PyTorch
- torchvision
- NLTK
- Pillow
- NumPy
- tqdm

## Usage

### Installation
```bash
pip install -r requirements.txt
```

## Future Improvements
- Implement attention mechanism
- Try transformer-based decoder instead of LSTM
- Experiment with different CNN backbones
- Fine-tune the entire model end-to-end
