# Image-Caption-Generator-CNN-VGG-16-RNN-LSTM-

![idea](https://github.com/user-attachments/assets/0d9eb1ad-0f47-4bcd-ae76-1f6a575d01bc)


## Overview
This project implements an **Image Caption Generator** using a Convolutional Neural Network (CNN) as an encoder and a Recurrent Neural Network (RNN) as a decoder. The encoder extracts features from input images using the pre-trained VGG16 model, and the decoder generates captions by predicting the next word in the sequence based on previous words. The model is trained and evaluated on the Flickr8k dataset, which contains images paired with their corresponding captions.

## Dataset
### Flickr8k Dataset
The dataset consists of:
- 8,000 images
- 40,000 captions (5 captions per image)

The dataset can be downloaded from Kaggle: [Flickr8k Dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k). Ensure the dataset is placed in the `BASE_DIR` directory specified in the code.

## Project Workflow
### 1. Feature Extraction with CNN
- A pre-trained **VGG16** model is used to extract features from input images.
- The model is modified by removing the fully connected layers and retaining the convolutional layers.
- Image features are preprocessed and stored for later use.

### 2. Text Preprocessing
- Tokenization is performed on the captions to convert words into integer sequences.
- A vocabulary is built based on the training data.
- Captions are padded to ensure uniform input size.

### 3. Building the Model
- **Encoder**: Extracts image features using the pre-trained VGG16 model.
- **Decoder**: Utilizes an embedding layer, an LSTM layer, and a dense layer to generate captions.
- The two components are combined using a functional API.
-![output123](https://github.com/user-attachments/assets/9528526d-58d3-4c2e-bc32-2aa4324c4f66)
 

### 4. Training
- The model is trained to predict the next word in a caption given the image and the previous words.
- Sparse categorical cross-entropy is used as the loss function.

### 5. Inference
- During inference, the decoder generates captions word-by-word for a given image.
- Beam search is implemented to improve caption quality.

### 6. Evaluation
- The BLEU (Bilingual Evaluation Understudy) score is used to evaluate the quality of the generated captions.

## Key Libraries Used
- `TensorFlow` / `Keras` for model building
- `NumPy` for numerical operations
- `Pillow` for image handling
- `tqdm` for progress visualization

## Instructions
### 1. Setup Environment
Install the required Python libraries:
```bash
pip install tensorflow numpy pillow tqdm
```

### 2. Clone Repository
```bash
git clone https://github.com/yourusername/image-caption-generator.git
cd image-caption-generator
```

### 3. Run the Notebook
Run the provided Jupyter Notebook to:
1. Extract features
2. Train the model
3. Generate captions for new images

### 4. Test with New Images
Place new images in the specified folder and generate captions using the trained model.

## Results
The model generates meaningful captions that align with the content of the input images. BLEU scores demonstrate the quality of captions when compared to ground truth.

## Future Work
- Explore other pre-trained models for feature extraction (e.g., ResNet, Inception).
- Implement attention mechanisms to improve caption accuracy.
- Fine-tune the decoder on domain-specific datasets.

## Acknowledgments
- [Kaggle: Flickr8k Dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k)
- [1]	S.-E.- Fatima, K. Gupta, D. Goyal, and S. K. Mishra, “Image caption generation using deep learning algorithm,” eatp, vol. 30, no. 5, pp. 8118–8128, 2024.


## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

