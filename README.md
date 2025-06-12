# Wikipedia Title Generation - NLP Assignment 2

A comprehensive implementation of sequence-to-sequence models for automatically generating Wikipedia article titles from article content. This project demonstrates the evolution from traditional RNN architectures to modern transformer models for the challenging task of extreme document summarization.

## Dataset

The dataset contains Wikipedia articles with the following structure:
- **Training set**: ~14,000 articles
- **Test set**: 100 articles  
- **Validation set**: 500 articles (extracted from training)
- **Format**: CSV files with `text` (article body) and `title` columns

**Dataset Statistics:**
- Average article length: ~2,500 words
- Average title length: 4-8 words
- Domain coverage: Diverse Wikipedia topics
- Language: English

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for transformer models)
- At least 8GB RAM (16GB recommended)

### Setup Instructions

```bash
# Clone the repository
git clone https://github.com/your-username/wikipedia-title-generator.git
cd wikipedia-title-generator

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data (automatically handled in code)
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Verify installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
```

### Requirements

```txt
torch>=1.9.0
transformers>=4.20.0
datasets>=2.0.0
pandas>=1.3.0
numpy>=1.21.0
nltk>=3.7
rouge>=1.0.1
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
tqdm>=4.62.0
```

## Usage

### Quick Start

```bash
# Run complete pipeline
python data_preprocessing.py    # Preprocess data
python rnn_seq2seq_models.py   # Train RNN models
python transformer_models.py   # Train transformer models
```

### Task A: Data Preprocessing

```bash
python data_preprocessing.py
```

**Features:**
- Extracts 500 validation samples from training data
- Comprehensive text cleaning (punctuation, non-ASCII removal)
- NLTK-based stopword removal and Porter stemming
- Generates both processed (RNN) and raw (transformer) datasets

**Output Files:**
```
train_processed.csv, val_processed.csv, test_processed.csv  # For RNN models
train_raw.csv, val_raw.csv, test_raw.csv                   # For transformers
```

### Task B: RNN-based Models

```bash
python rnn_seq2seq_models.py
```

**Available Models:**
1. **Basic RNN**: Bidirectional encoder + GRU decoder
2. **GloVe Enhanced**: Pre-trained embeddings integration
3. **Hierarchical**: Word and sentence-level encoding
4. **Multi-layer Decoder**: Two-layer GRU decoder

**Training Configuration:**
```python
# Model hyperparameters
embedding_dim = 300
hidden_dim = 300
batch_size = 32
learning_rate = 0.001
epochs = 10
```

### Task C: Transformer Models

```bash
python transformer_models.py
```

**C1: Fine-tuning T5-small**
- Hyperparameter search across multiple configurations
- Both greedy and beam search evaluation
- Automatic best model selection

**C2: Zero-shot Prompting**
- Flan-T5-base and Flan-T5-large models
- Multiple prompt engineering strategies
- No training required

## Model Architectures

### RNN-based Models

#### Basic Encoder-Decoder
```python
class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)
```

**Key Features:**
- Bidirectional GRU encoder (300 hidden units)
- Attention mechanism between encoder-decoder
- Teacher forcing with 50% ratio
- Gradient clipping and dropout regularization

#### Enhanced Variants

**1. Hierarchical Encoder**
- **Word-level**: Bidirectional GRU for token sequences
- **Sentence-level**: Bidirectional GRU for sentence representations
- Two-tier document understanding

**2. Multi-layer Decoder**
- Two-layer GRU architecture
- Enhanced sequence generation capability
- Better long-term dependency modeling

**3. Beam Search Decoding**
- Explores multiple generation paths
- Beam width: 3 for optimal performance
- Improved output quality over greedy search

### Transformer Models

#### T5-small Architecture
```python
# Model: google-t5/t5-small (60M parameters)
# Input: "summarize: " + article_text
# Output: Generated title
# Max input length: 512 tokens
# Max output length: 64 tokens
```

#### Flan-T5 Models
```python
# Flan-T5-base: 250M parameters
# Flan-T5-large: 780M parameters
# Instruction-tuned for zero-shot performance
```

## Performance Results

### Task B: RNN-based Models

| Model Architecture | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 | Training Time |
|-------------------|-------------|-------------|-------------|---------------|
| Basic RNN | 0.5802 | 0.2150 | 0.5802 | 458.21s |
| RNN + Beam Search | 0.5846 | 0.2250 | 0.5846 | - |
| RNN + GloVe | 0.5723 | 0.2335 | 0.5723 | 460.03s |
| RNN + Hierarchical | 0.5722 | 0.2223 | 0.5722 | 4668.89s |
| **RNN + 2-Layer Decoder** | **0.6214** | **0.2787** | **0.6145** | 758.59s |

### Task C: Transformer Models

#### Fine-tuned T5-small
| Decoding Strategy | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 | Time |
|------------------|-------------|-------------|-------------|------|
| **Greedy Search** | **0.8860** | **0.6628** | **0.8860** | 2.02s |
| **Beam Search** | **0.8960** | **0.6828** | **0.8960** | 4.02s |

#### Zero-shot Prompting Results

**Flan-T5-base (250M parameters):**
| Prompt Strategy | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 |
|----------------|-------------|-------------|-------------|
| "Generate a title for this Wikipedia article" | **0.7844** | **0.5689** | **0.7844** |
| "Create a concise, informative title..." | 0.4055 | 0.2319 | 0.4039 |
| "Summarize this Wikipedia article..." | 0.4533 | 0.2549 | 0.4533 |
| "Extract the main topic and create..." | 0.7428 | 0.5729 | 0.7372 |

**Flan-T5-large (780M parameters):**
| Prompt Strategy | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 |
|----------------|-------------|-------------|-------------|
| "Generate a title for this Wikipedia article" | **0.8580** | **0.6400** | **0.8580** |
| "Create a concise, informative title..." | 0.6340 | 0.4731 | 0.6324 |
| "Summarize this Wikipedia article..." | 0.7628 | 0.5326 | 0.7612 |
| "Extract the main topic and create..." | 0.8320 | 0.6373 | 0.8320 |

## Key Findings

### Performance Insights

1. **Transformer Superiority**: Fine-tuned T5-small achieved **38% better performance** than the best RNN model (ROUGE-1: 0.8960 vs 0.6214)

2. **Zero-shot Effectiveness**: Flan-T5-large with optimal prompting achieved comparable performance to fine-tuned models without task-specific training

3. **Prompt Engineering Impact**: Prompt wording dramatically affected performance - best vs worst prompts showed nearly **2x difference** in ROUGE scores

4. **Architectural Improvements**: Among RNN variants, the 2-layer decoder provided the most significant improvement over the basic model

### Computational Efficiency

- **RNN Training**: 458-4669 seconds depending on architecture
- **Transformer Fine-tuning**: 10,606 seconds for comprehensive hyperparameter search
- **Zero-shot Inference**: 32-458 seconds (no training required)

### Model Selection Guidelines

1. **High Performance Required**: Use fine-tuned T5-small or larger transformer models
2. **Limited Computational Resources**: 2-layer decoder RNN provides reasonable compromise
3. **No Training Data Available**: Flan-T5 with carefully engineered prompts
4. **Real-time Applications**: Consider inference time vs quality tradeoffs

## Technical Implementation

### Advanced Features

#### Vocabulary Management
```python
class Vocabulary:
    def __init__(self, min_freq_percent=1.0):
        # Frequency-based filtering
        # Special tokens: PAD, SOS, EOS, UNK
        # Dynamic vocabulary construction
```

#### Beam Search Implementation
```python
def _beam_search(self, encoder_hidden, batch_size, beam_width, max_length):
    # Maintains multiple hypotheses
    # Ranks by accumulated log probability
    # Configurable beam width
```

#### Hyperparameter Optimization
```python
hyperparameter_configs = [
    {"learning_rate": 1e-4, "batch_size": 4, "epochs": 3},
    {"learning_rate": 5e-5, "batch_size": 8, "epochs": 2},
    # ... more configurations
]
```

### Error Handling and Robustness

- **CUDA Availability**: Automatic GPU/CPU detection
- **Memory Management**: Batch processing for large datasets
- **Exception Handling**: Robust error handling for evaluation
- **Input Validation**: Safeguards against empty or malformed inputs

## Examples

### Generated Title Examples

**Input Article (excerpt):**
> "Machine learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence based on the idea that systems can learn from data..."

**Generated Titles:**
- **T5-small (fine-tuned)**: "Machine learning"
- **Flan-T5-large**: "Machine Learning"
- **RNN (2-layer)**: "machine learn"
- **Reference**: "Machine learning"

### Prompt Engineering Examples

```python
# Best performing prompt
"Generate a title for this Wikipedia article:"

# Poor performing prompt  
"Create a concise, informative title for the following text:"

# Performance difference: ~2x ROUGE score improvement
```
