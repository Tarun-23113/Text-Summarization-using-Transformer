# Text Summarization using Encoder-Decoder Transformer

A text summarization application using the pre-trained BART model from Hugging Face.

## Features

- **BART Model**: Uses Facebook's BART-large-CNN model for abstractive summarization
- **Interactive UI**: Built with Streamlit for easy interaction
- **Customizable**: Adjust summary length parameters
- **Real-time Statistics**: Shows compression ratio and word counts

## Architecture

This project implements an **Encoder-Decoder Transformer** architecture:

- **Encoder**: Processes input text and creates contextual embeddings using self-attention
- **Decoder**: Generates summary using cross-attention to encoder outputs
- **Attention Mechanisms**: Allows model to focus on relevant input parts
- **Tokenization**: Converts text to tokens using BART's tokenizer

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Local Development

Run the Streamlit app:

```bash
streamlit run main.py
```

The app will open in your browser at `http://localhost:8501`

### Deployment on Streamlit Cloud

This app is deployed on Streamlit Cloud. Visit the live demo at: [Your App URL]

## How It Works

1. **Input**: User provides text to summarize
2. **Tokenization**: Text is converted to tokens
3. **Encoding**: Encoder processes tokens and creates embeddings
4. **Decoding**: Decoder generates summary tokens using attention
5. **Output**: Tokens are converted back to readable text

## Model Details

- **Model**: `facebook/bart-large-cnn`
- **Type**: Encoder-Decoder Transformer
- **Training**: Pre-trained on CNN/DailyMail dataset
- **Task**: Abstractive text summarization

## Technologies

- Hugging Face Transformers
- PyTorch
- Streamlit
- BART (Bidirectional and Auto-Regressive Transformers)
