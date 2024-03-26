# Character-Level Decoder Transformer Model for Script Generation

This repository contains code for training a character-level decoder only transformer model on a script dataset from the TV show "The Office".

## Dataset
The dataset used in this project is a script from "The Office" TV show. The script is assumed to be stored in a file named `output.txt`. 

## Model
The model used is a character-level decoder only transformer. It generates text character by character, which makes it suitable for text generation tasks.
The provided code implements a character-level language model based on a Transformer architecture. It comprises components such as Scaled Dot-Product Attention and 
Multi-Head Attention, enabling the model to capture intricate dependencies within the input text. Additionally, a Feed Forward Network aids in capturing higher-level abstractions. 
The model consists of Transformer blocks, each containing attention and feed-forward layers with layer normalization. During training, cross-entropy loss is utilized, 
while during generation, tokens are sampled based on predicted probabilities. This lightweight model is designed for character-level text generation tasks,
such as script generation.


## Training
The provided code is for training a character-level language model using a Transformer architecture. It begins by initializing parameters and creating directories to save 
model checkpoints and generated text. The script dataset from "The Office" TV show is loaded, and characters are encoded. 
Training involves iterations where batches of input data are processed. 
Validation loss is monitored, and the best model is saved based on validation performance. 
The model is trained using the AdamW optimizer with backpropagation. 
Throughout training, the model's performance is evaluated, and the best model is saved for later use in text generation tasks.

## Deploying the app on streamlit
Finally we load the saved model that we saved during training and then use streamlit to deploy out app. Now the app can generate scripts for any given prompt. 
The working of the app is as follows



https://github.com/mishra-kunal1/The-Office-Script-Generator-using-LLM/assets/99056351/5d480725-d7fb-4f15-80bb-46d9abb5ac76

