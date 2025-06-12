import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from rouge import Rouge
import random
import math

# Special tokens
PAD_token = 0
SOS_token = 1
EOS_token = 2
UNK_token = 3

class Vocabulary:
    def __init__(self, min_freq_percent=1.0):
        self.word2index = {"<pad>": PAD_token, "<bos>": SOS_token, "<eos>": EOS_token, "<unk>": UNK_token}
        self.word2count = {}
        self.index2word = {PAD_token: "<pad>", SOS_token: "<bos>", EOS_token: "<eos>", UNK_token: "<unk>"}
        self.n_words = 4  # Count default tokens
        self.min_freq_percent = min_freq_percent
        
    def add_sentence(self, sentence):
        if isinstance(sentence, str):
            for word in sentence.split():
                self.add_word(word)

            
    def add_word(self, word):
        if word not in self.word2count:
            self.word2count[word] = 1
        else:
            self.word2count[word] += 1
    
    def build_vocab(self, total_sentences):
        min_freq = math.ceil(total_sentences * self.min_freq_percent / 100)
        for word, count in self.word2count.items():
            if count >= min_freq:
                self.word2index[word] = self.n_words
                self.index2word[self.n_words] = word
                self.n_words += 1

class WikiTitleDataset(Dataset):
    def __init__(self, dataframe, vocab, max_length=512):
        self.dataframe = dataframe
        self.vocab = vocab
        self.max_length = max_length
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        text = self.dataframe.iloc[idx]['text']
        title = self.dataframe.iloc[idx]['title']
        
        # Convert text and title to indices
        text_indices = self.tokenize(text)
        title_indices = self.tokenize(title)
        
        # Add SOS and EOS tokens to title
        title_indices = [SOS_token] + title_indices + [EOS_token]
        
        return {
            'text': torch.tensor(text_indices, dtype=torch.long),
            'title': torch.tensor(title_indices, dtype=torch.long),
            'text_length': len(text_indices),
            'title_length': len(title_indices)
        }
    
    def tokenize(self, sentence):
        indices = []
        if isinstance(sentence, str):
            for word in sentence.split()[:self.max_length]:
                if word in self.vocab.word2index:
                    indices.append(self.vocab.word2index[word])
                else:
                    indices.append(UNK_token)
        return indices


def collate_fn(batch):
    # Sort by text length (descending)
    batch.sort(key=lambda x: x['text_length'], reverse=True)
    
    # Get max lengths
    max_text_length = max([item['text_length'] for item in batch])
    max_title_length = max([item['title_length'] for item in batch])
    
    # Pad sequences
    text_padded = torch.zeros(len(batch), max_text_length, dtype=torch.long)
    title_padded = torch.zeros(len(batch), max_title_length, dtype=torch.long)
    
    for i, item in enumerate(batch):
        text_padded[i, :item['text_length']] = item['text']
        title_padded[i, :item['title_length']] = item['title']
    
    return {
        'text': text_padded,
        'title': title_padded,
        'text_length': torch.tensor([item['text_length'] for item in batch]),
        'title_length': torch.tensor([item['title_length'] for item in batch])
    }

class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout=0.5):
        super(EncoderRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_seq):
        # input_seq: [batch_size, seq_len]
        
        embedded = self.dropout(self.embedding(input_seq))
        # embedded: [batch_size, seq_len, embedding_dim]
        
        outputs, hidden = self.gru(embedded)
        # outputs: [batch_size, seq_len, hidden_dim * 2]
        # hidden: [2, batch_size, hidden_dim]
        
        # Combine bidirectional hidden states
        hidden = torch.cat((hidden[0], hidden[1]), dim=1)
        # hidden: [batch_size, hidden_dim * 2]
        
        hidden = self.fc(hidden)
        # hidden: [batch_size, hidden_dim]
        
        return outputs, hidden
    
    def load_embeddings(self, embeddings_path, vocab):
        """Load pre-trained GloVe embeddings"""
        embeddings = {}
        with open(embeddings_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = torch.tensor([float(val) for val in values[1:]])
                embeddings[word] = vector
                
        # Initialize embedding weights
        embedding_dim = self.embedding.weight.size(1)
        for word, idx in vocab.word2index.items():
            if word in embeddings:
                self.embedding.weight.data[idx] = embeddings[word]

class DecoderRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout=0.5):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_token, hidden):
        # input_token: [batch_size]
        # hidden: [batch_size, hidden_dim]
        
        if input_token.dim() == 0:
            input_token = input_token.unsqueeze(0)
        embedded = self.dropout(self.embedding(input_token.unsqueeze(1)))

        # embedded: [batch_size, 1, embedding_dim]
        
        if hidden.dim() == 2:
            hidden = hidden.unsqueeze(0)  # Add num_layers dimension if missing
        output, hidden = self.gru(embedded, hidden)
        # output: [batch_size, 1, hidden_dim]
        # hidden: [1, batch_size, hidden_dim]
        
        prediction = self.fc(output.squeeze(1))
        # prediction: [batch_size, vocab_size]
        
        return prediction, hidden.squeeze(0)

class Decoder2RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout=0.5):
        super(Decoder2RNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru1 = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.gru2 = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_token, hidden):
        # input_token: [batch_size]
        # hidden: [batch_size, hidden_dim]
        
        embedded = self.dropout(self.embedding(input_token.unsqueeze(1)))
        # embedded: [batch_size, 1, embedding_dim]
        
        # Split hidden state for the two GRUs
        hidden1 = hidden.unsqueeze(0)
        hidden2 = hidden.unsqueeze(0)
        
        output1, hidden1 = self.gru1(embedded, hidden1)
        # output1: [batch_size, 1, hidden_dim]
        
        output2, hidden2 = self.gru2(output1, hidden2)
        # output2: [batch_size, 1, hidden_dim]
        
        prediction = self.fc(output2.squeeze(1))
        # prediction: [batch_size, vocab_size]
        
        return prediction, hidden2.squeeze(0)

class HierEncoderRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout=0.5):
        super(HierEncoderRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.word_gru = nn.GRU(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.sent_gru = nn.GRU(hidden_dim * 2, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_seq, sentence_boundaries=None):
        # input_seq: [batch_size, seq_len]
        # sentence_boundaries: list of lists containing sentence end indices
        
        batch_size = input_seq.size(0)
        
        embedded = self.dropout(self.embedding(input_seq))
        # embedded: [batch_size, seq_len, embedding_dim]
        
        # Word-level encoding
        word_outputs, word_hidden = self.word_gru(embedded)
        # word_outputs: [batch_size, seq_len, hidden_dim * 2]
        
        # If no sentence boundaries provided, split by fixed length
        if sentence_boundaries is None:
            sentence_boundaries = [list(range(0, input_seq.size(1), 20)) for _ in range(batch_size)]
            if sentence_boundaries[0][-1] != input_seq.size(1) - 1:
                for i in range(batch_size):
                    sentence_boundaries[i].append(input_seq.size(1) - 1)
        
        # Prepare sentence representations
        max_sentences = max([len(boundaries) for boundaries in sentence_boundaries])
        sent_representations = torch.zeros(batch_size, max_sentences, word_outputs.size(2), device=input_seq.device)
        
        for b in range(batch_size):
            for i, boundary in enumerate(sentence_boundaries[b]):
                if i == 0:
                    start_idx = 0
                else:
                    start_idx = sentence_boundaries[b][i-1] + 1
                end_idx = boundary + 1
                
                # Average word representations for this sentence
                sent_representations[b, i] = word_outputs[b, start_idx:end_idx].mean(dim=0)
        
        # Sentence-level encoding
        sent_outputs, sent_hidden = self.sent_gru(sent_representations)
        # sent_outputs: [batch_size, max_sentences, hidden_dim * 2]
        # sent_hidden: [2, batch_size, hidden_dim]
        
        # Combine bidirectional hidden states
        hidden = torch.cat((sent_hidden[0], sent_hidden[1]), dim=1)
        # hidden: [batch_size, hidden_dim * 2]
        
        hidden = self.fc(hidden)
        # hidden: [batch_size, hidden_dim]
        
        return sent_outputs, hidden
    
    def load_embeddings(self, embeddings_path):
        """Load pre-trained GloVe embeddings"""
        embeddings = {}
        with open(embeddings_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = torch.tensor([float(val) for val in values[1:]])
                embeddings[word] = vector
                
        # Initialize embedding weights
        embedding_dim = self.embedding.weight.size(1)
        for word, idx in self.vocab.word2index.items():
            if word in embeddings:
                self.embedding.weight.data[idx] = embeddings[word]

class Seq2seqRNN(nn.Module):
    def __init__(self, encoder, decoder, device, max_length=100, use_hierarchical=False, use_decoder2=False):
        super(Seq2seqRNN, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.max_length = max_length
        self.use_hierarchical = use_hierarchical
        
    def forward(self, src, tgt=None, teacher_forcing_ratio=0.5, max_new_tokens=100, use_beam_search=False, beam_width=3):
        # src: [batch_size, src_len]
        # tgt: [batch_size, tgt_len] or None during inference
        
        batch_size = src.size(0)
        
        # Encode the source sequence
        encoder_outputs, encoder_hidden = self.encoder(src)
        
        # Initialize decoder input and hidden state
        decoder_input = torch.tensor([SOS_token] * batch_size, device=self.device)
        decoder_hidden = encoder_hidden
        
        # Prepare output tensor
        if tgt is not None:
            target_length = tgt.size(1)
            outputs = torch.zeros(batch_size, target_length, self.decoder.fc.out_features, device=self.device)
        else:
            target_length = max_new_tokens
            outputs = torch.zeros(batch_size, target_length, self.decoder.fc.out_features, device=self.device)
        
        # Store generated tokens
        generated_tokens = torch.zeros(batch_size, target_length, dtype=torch.long, device=self.device)
        
        # Teacher forcing: use target as the next input
        use_teacher_forcing = True if tgt is not None and random.random() < teacher_forcing_ratio else False
        
        if use_beam_search and tgt is None:
            # Beam search for inference
            return self._beam_search(encoder_hidden, batch_size, beam_width, max_new_tokens)
        
        # Greedy decoding
        for t in range(target_length):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            outputs[:, t] = decoder_output
            
            # Get the most likely token
            top1 = decoder_output.argmax(1)
            generated_tokens[:, t] = top1
            
            # Next input is either from teacher forcing or the predicted token
            if use_teacher_forcing and t < tgt.size(1) - 1:
                decoder_input = tgt[:, t+1]
            else:
                decoder_input = top1
            
            # Stop if all sequences have generated EOS token
            if tgt is None and (top1 == EOS_token).all():
                break
        
        return outputs, generated_tokens
    
    def _beam_search(self, encoder_hidden, batch_size, beam_width, max_length):
        """
        Perform beam search decoding
        """
        # Initialize beam for each batch item
        beams = [[(torch.tensor([SOS_token], device=self.device), 0.0, encoder_hidden[b:b+1])] 
        for b in range(batch_size)]
        
        completed_beams = [[] for _ in range(batch_size)]
        
        # Beam search for each batch item
        for _ in range(max_length):
            for b in range(batch_size):
                if len(beams[b]) == 0:  # Skip if all beams are completed
                    continue
                
                # Collect all candidates from current beams
                candidates = []
                for seq, score, hidden in beams[b]:
                    if seq[-1].item() == EOS_token:
                        # Add completed sequence to results
                        completed_beams[b].append((seq, score, hidden))
                        continue
                    
                    # Predict next token probabilities
                    decoder_input = seq[-1]
                    decoder_output, new_hidden = self.decoder(decoder_input, hidden)
                    
                    # Get top k tokens
                    log_probs, indices = torch.topk(torch.log_softmax(decoder_output, dim=1), beam_width)
                    
                    for i in range(beam_width):
                        new_seq = torch.cat([seq, indices[0, i].unsqueeze(0)])
                        new_score = score + log_probs[0, i].item()
                        candidates.append((new_seq, new_score, new_hidden.unsqueeze(0)))
                
                # Select top k candidates
                candidates.sort(key=lambda x: x[1], reverse=True)
                beams[b] = candidates[:beam_width]
            
            # Check if all beams are completed
            if all(len(beam) == 0 for beam in beams):
                break
        
        # Collect final results
        results = []
        for b in range(batch_size):
            # Combine completed beams and remaining beams
            all_beams = completed_beams[b] + beams[b]
            all_beams.sort(key=lambda x: x[1], reverse=True)
            
            # Take the best beam
            best_seq = all_beams[0][0] if all_beams else torch.tensor([SOS_token, EOS_token], device=self.device)
            results.append(best_seq)
        
        # Pad sequences to same length
        max_len = max([len(seq) for seq in results])
        padded_results = torch.zeros(batch_size, max_len, dtype=torch.long, device=self.device)
        
        for b, seq in enumerate(results):
            padded_results[b, :len(seq)] = seq
        
        return padded_results

def train_epoch(model, dataloader, optimizer, criterion, device, clip=1.0):
    model.train()
    epoch_loss = 0
    
    for batch in dataloader:
        src = batch['text'].to(device)
        tgt = batch['title'].to(device)
        
        optimizer.zero_grad()
        
        output, _ = model(src, tgt, teacher_forcing_ratio=0.5)
        
        # Reshape output and target for loss calculation
                # Reshape output and target for loss calculation
        output_dim = output.shape[-1]
        output = output[:, :-1].contiguous().view(-1, output_dim)
        tgt = tgt[:, 1:].contiguous().view(-1)
        
        # Calculate loss
        loss = criterion(output, tgt)
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        # Update parameters
        optimizer.step()
        
        epoch_loss += loss.item()
    
    return epoch_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for batch in dataloader:
            src = batch['text'].to(device)
            tgt = batch['title'].to(device)
            
            output, _ = model(src, tgt, teacher_forcing_ratio=0.0)
            
            # Reshape output and target for loss calculation
            output_dim = output.shape[-1]
            output = output[:, :-1].contiguous().view(-1, output_dim)
            tgt = tgt[:, 1:].contiguous().view(-1)
            
            # Calculate loss
            loss = criterion(output, tgt)
            
            epoch_loss += loss.item()
    
    return epoch_loss / len(dataloader)

def generate_titles(model, dataloader, vocab, device, max_length=100, use_beam_search=False, beam_width=3):
    model.eval()
    generated_titles = []
    reference_titles = []
    
    with torch.no_grad():
        for batch in dataloader:
            src = batch['text'].to(device)
            tgt = batch['title']
            
            if use_beam_search:
                output = model(src, tgt=None, teacher_forcing_ratio=0.0, max_new_tokens=max_length, use_beam_search=True, beam_width=beam_width)
            else:
                _, output = model(src, tgt=None, teacher_forcing_ratio=0.0, max_new_tokens=max_length)
            
            # Convert indices to words
            for i in range(output.size(0)):
                title_tokens = []
                for token_idx in output[i]:
                    if token_idx.item() == EOS_token:
                        break
                    if token_idx.item() not in [PAD_token, SOS_token]:
                        title_tokens.append(vocab.index2word[token_idx.item()])
                
                generated_title = ' '.join(title_tokens)
                generated_titles.append(generated_title)
                
                # Get reference title
                ref_tokens = []
                for token_idx in tgt[i]:
                    if token_idx.item() == EOS_token:
                        break
                    if token_idx.item() not in [PAD_token, SOS_token]:
                        ref_tokens.append(vocab.index2word[token_idx.item()])
                
                reference_title = ' '.join(ref_tokens)
                reference_titles.append(reference_title)
    
    return generated_titles, reference_titles

def calculate_rouge_scores(generated_titles, reference_titles):
    rouge = Rouge()
    scores = rouge.get_scores(generated_titles, reference_titles, avg=True)
    return scores

def main():
    import time
    start_time_total = time.time()
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load preprocessed data
    start_time = time.time()
    train_df = pd.read_csv('train_processed.csv')
    val_df = pd.read_csv('val_processed.csv')
    test_df = pd.read_csv('test_processed.csv')
    print(f"Time taken to load data: {time.time() - start_time:.2f} seconds")

    # Build vocabulary
    start_time = time.time()
    vocab = Vocabulary(min_freq_percent=1.0)
    for text in train_df['text']:
        vocab.add_sentence(text)
    for title in train_df['title']:
        vocab.add_sentence(title)
    vocab.build_vocab(len(train_df))
    print(f"Vocabulary size: {vocab.n_words}")
    print(f"Time taken to build vocabulary: {time.time() - start_time:.2f} seconds")

    # Create datasets
    start_time = time.time()
    train_dataset = WikiTitleDataset(train_df, vocab)
    val_dataset = WikiTitleDataset(val_df, vocab)
    test_dataset = WikiTitleDataset(test_df, vocab)
    print(f"Time taken to create datasets: {time.time() - start_time:.2f} seconds")

    # Create dataloaders
    start_time = time.time()
    batch_size = 32
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)
    print(f"Time taken to create dataloaders: {time.time() - start_time:.2f} seconds")

    # Model hyperparameters
    embedding_dim = 300
    hidden_dim = 300
    dropout = 0.5

    # Initialize model
    start_time = time.time()
    encoder = EncoderRNN(vocab.n_words, embedding_dim, hidden_dim, dropout)
    decoder = DecoderRNN(vocab.n_words, embedding_dim, hidden_dim, dropout)
    model = Seq2seqRNN(encoder, decoder, device)
    model = model.to(device)
    print(f"Time taken to initialize model: {time.time() - start_time:.2f} seconds")

    # Initialize optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_token)

    # Training loop
    start_time = time.time()
    n_epochs = 10
    best_val_loss = float('inf')
    for epoch in range(n_epochs):
        epoch_start_time = time.time()
        train_loss = train_epoch(model, train_dataloader, optimizer, criterion, device)
        val_loss = evaluate(model, val_dataloader, criterion, device)
        print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Time taken for epoch {epoch+1}: {time.time() - epoch_start_time:.2f} seconds")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pt')
            print(f"Model saved with validation loss: {val_loss:.4f}")
    print(f"Total training time: {time.time() - start_time:.2f} seconds")

    # Load best model for evaluation
    start_time = time.time()
    model.load_state_dict(torch.load('best_model.pt'))
    print(f"Time taken to load best model: {time.time() - start_time:.2f} seconds")

    # Generate titles and calculate metrics
    start_time = time.time()
    generated_titles, reference_titles = generate_titles(model, test_dataloader, vocab, device)
    print(f"Time taken to generate titles: {time.time() - start_time:.2f} seconds")

    # Calculate ROUGE scores
    start_time = time.time()
    rouge_scores = calculate_rouge_scores(generated_titles, reference_titles)
    print(f"ROUGE-1 F1: {rouge_scores['rouge-1']['f']:.4f}")
    print(f"ROUGE-2 F1: {rouge_scores['rouge-2']['f']:.4f}")
    print(f"ROUGE-L F1: {rouge_scores['rouge-l']['f']:.4f}")
    print(f"Time taken to calculate ROUGE scores: {time.time() - start_time:.2f} seconds")

    # Test with beam search
    print("\nGenerating titles with beam search...")
    start_time = time.time()
    beam_titles, reference_titles = generate_titles(model, test_dataloader, vocab, device, use_beam_search=True, beam_width=3)
    print(f"Time taken for beam search generation: {time.time() - start_time:.2f} seconds")

    # Calculate ROUGE scores for beam search
    start_time = time.time()
    beam_rouge_scores = calculate_rouge_scores(beam_titles, reference_titles)
    print(f"Beam Search ROUGE-1 F1: {beam_rouge_scores['rouge-1']['f']:.4f}")
    print(f"Beam Search ROUGE-2 F1: {beam_rouge_scores['rouge-2']['f']:.4f}")
    print(f"Beam Search ROUGE-L F1: {beam_rouge_scores['rouge-l']['f']:.4f}")
    print(f"Time taken to calculate beam search ROUGE scores: {time.time() - start_time:.2f} seconds")

    # Try with GloVe embeddings
    print("\nTraining model with GloVe embeddings...")
    start_time = time.time()
    encoder_glove = EncoderRNN(vocab.n_words, embedding_dim, hidden_dim, dropout)
    encoder_glove.load_embeddings('glove.6B.300d.txt', vocab)
    model_glove = Seq2seqRNN(encoder_glove, decoder, device)
    model_glove = model_glove.to(device)
    print(f"Time taken to initialize GloVe model: {time.time() - start_time:.2f} seconds")

    # Initialize optimizer and criterion
    optimizer_glove = optim.Adam(model_glove.parameters(), lr=0.001)

    # Training loop for GloVe model
    start_time = time.time()
    for epoch in range(n_epochs):
        epoch_start_time = time.time()
        train_loss = train_epoch(model_glove, train_dataloader, optimizer_glove, criterion, device)
        val_loss = evaluate(model_glove, val_dataloader, criterion, device)
        print(f"GloVe Epoch {epoch+1}/{n_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Time taken for GloVe epoch {epoch+1}: {time.time() - epoch_start_time:.2f} seconds")
    print(f"Total GloVe training time: {time.time() - start_time:.2f} seconds")

    # Generate titles with GloVe model
    start_time = time.time()
    glove_titles, reference_titles = generate_titles(model_glove, test_dataloader, vocab, device)
    print(f"Time taken to generate GloVe titles: {time.time() - start_time:.2f} seconds")

    # Calculate ROUGE scores for GloVe model
    start_time = time.time()
    glove_rouge_scores = calculate_rouge_scores(glove_titles, reference_titles)
    print(f"GloVe ROUGE-1 F1: {glove_rouge_scores['rouge-1']['f']:.4f}")
    print(f"GloVe ROUGE-2 F1: {glove_rouge_scores['rouge-2']['f']:.4f}")
    print(f"GloVe ROUGE-L F1: {glove_rouge_scores['rouge-l']['f']:.4f}")
    print(f"Time taken to calculate GloVe ROUGE scores: {time.time() - start_time:.2f} seconds")

    # Try with hierarchical encoder
    print("\nTraining model with hierarchical encoder...")
    start_time = time.time()
    hier_encoder = HierEncoderRNN(vocab.n_words, embedding_dim, hidden_dim, dropout)
    model_hier = Seq2seqRNN(hier_encoder, decoder, device, use_hierarchical=True)
    model_hier = model_hier.to(device)
    print(f"Time taken to initialize hierarchical model: {time.time() - start_time:.2f} seconds")

    # Initialize optimizer and criterion
    optimizer_hier = optim.Adam(model_hier.parameters(), lr=0.001)

    # Training loop for hierarchical model
    start_time = time.time()
    for epoch in range(n_epochs):
        epoch_start_time = time.time()
        train_loss = train_epoch(model_hier, train_dataloader, optimizer_hier, criterion, device)
        val_loss = evaluate(model_hier, val_dataloader, criterion, device)
        print(f"Hierarchical Epoch {epoch+1}/{n_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Time taken for hierarchical epoch {epoch+1}: {time.time() - epoch_start_time:.2f} seconds")
    print(f"Total hierarchical training time: {time.time() - start_time:.2f} seconds")

    # Generate titles with hierarchical model
    start_time = time.time()
    hier_titles, reference_titles = generate_titles(model_hier, test_dataloader, vocab, device)
    print(f"Time taken to generate hierarchical titles: {time.time() - start_time:.2f} seconds")

    # Calculate ROUGE scores for hierarchical model
    start_time = time.time()
    hier_rouge_scores = calculate_rouge_scores(hier_titles, reference_titles)
    print(f"Hierarchical ROUGE-1 F1: {hier_rouge_scores['rouge-1']['f']:.4f}")
    print(f"Hierarchical ROUGE-2 F1: {hier_rouge_scores['rouge-2']['f']:.4f}")
    print(f"Hierarchical ROUGE-L F1: {hier_rouge_scores['rouge-l']['f']:.4f}")
    print(f"Time taken to calculate hierarchical ROUGE scores: {time.time() - start_time:.2f} seconds")

    # Try with 2-layer decoder
    print("\nTraining model with 2-layer decoder...")
    start_time = time.time()
    decoder2 = Decoder2RNN(vocab.n_words, embedding_dim, hidden_dim, dropout)
    model_decoder2 = Seq2seqRNN(encoder, decoder2, device, use_decoder2=True)
    model_decoder2 = model_decoder2.to(device)
    print(f"Time taken to initialize 2-layer decoder model: {time.time() - start_time:.2f} seconds")

    # Initialize optimizer and criterion
    optimizer_decoder2 = optim.Adam(model_decoder2.parameters(), lr=0.001)

    # Training loop for 2-layer decoder model
    start_time = time.time()
    for epoch in range(n_epochs):
        epoch_start_time = time.time()
        train_loss = train_epoch(model_decoder2, train_dataloader, optimizer_decoder2, criterion, device)
        val_loss = evaluate(model_decoder2, val_dataloader, criterion, device)
        print(f"2-Layer Decoder Epoch {epoch+1}/{n_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Time taken for 2-layer decoder epoch {epoch+1}: {time.time() - epoch_start_time:.2f} seconds")
    print(f"Total 2-layer decoder training time: {time.time() - start_time:.2f} seconds")

    # Generate titles with 2-layer decoder model
    start_time = time.time()
    decoder2_titles, reference_titles = generate_titles(model_decoder2, test_dataloader, vocab, device)
    print(f"Time taken to generate 2-layer decoder titles: {time.time() - start_time:.2f} seconds")

    # Calculate ROUGE scores for 2-layer decoder model
    start_time = time.time()
    decoder2_rouge_scores = calculate_rouge_scores(decoder2_titles, reference_titles)
    print(f"2-Layer Decoder ROUGE-1 F1: {decoder2_rouge_scores['rouge-1']['f']:.4f}")
    print(f"2-Layer Decoder ROUGE-2 F1: {decoder2_rouge_scores['rouge-2']['f']:.4f}")
    print(f"2-Layer Decoder ROUGE-L F1: {decoder2_rouge_scores['rouge-l']['f']:.4f}")
    print(f"Time taken to calculate 2-layer decoder ROUGE scores: {time.time() - start_time:.2f} seconds")
    
    # Report total execution time
    print(f"\nTotal execution time: {time.time() - start_time_total:.2f} seconds")


if __name__ == "__main__":
    main()

