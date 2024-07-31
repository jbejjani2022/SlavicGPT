import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(3)

# path to data file
data_path = 'data/tiny-russian-lit/very_clean_tiny_russian_lit.txt'

with open(data_path, 'r', encoding='utf-8') as f:
    text = f.read()

# find the unique characters that occur in the text
chars = set(text)
vocab = ''.join(chars)
vocab_size = len(chars)

# create a simple character-level tokenizer: a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


# encoder: convert string to list of integers
def encode(s): return [stoi[c] for c in s]
# decoder: convert list of integers to string
def decode(l): return ''.join([itos[i] for i in l])


data = torch.tensor(encode(text), dtype=torch.long)

# split data into train and validation sets to test for overfitting
split = 0.8
n = int(split*len(data))
train_data = data[:n]
val_data = data[n:]

batch_size = 4  # the number of independent sequences that we will process in parallel
block_size = 8  # maximum context length for predictions


def get_batch(split):
    # generate a batch of data consisting of inputs x and targets y
    data = train_data if split == 'train' else val_data
    # generate batch_size random offsets in the interval [0, len(data) - batch_size)
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y


xb, yb = get_batch('train')


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # each token reads off the logits (input to softmax) for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B, T) tensor of integers (B = # batches, T = # timesteps/block size)
        # we are essentially predicting the next character based on the embedding of a single token
        # (B, T, C) : batch, time, channels
        logits = self.token_embedding_table(idx)

        if targets is None:
            loss = None
        else:
            # reshape logits since cross_entropy expects (B, C, T) inputs
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)  # equivalently, targets.view(-1)

            # negative log likelihood loss - calculates quality of our logits with respect to the true targets
            # a 'good' logit will have a high value in the target dimension and low values in other dimensions
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        # the bigram only uses the last char as the context
        # we pass in the full context here as practice for generation using transformer
        for _ in range(max_new_tokens):
            # get predictions
            logits, loss = self(idx)  # calls the forward function
            # retrieve only final timestep
            logits = logits[:, -1, :]  # (B, T, C) -> (B, C)
            # apply softmax to get probability distribution
            dist = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(dist, num_samples=1)  # (B, 1)
            # append new sample to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T + 1)
        return idx

    # sample text by generating new tokens and decoding to characters
    # by default, the context used to generate the first new token is a newline char
    def sample_text(self, context=torch.tensor([encode('\n')]), new_tokens=250):
        print(f'Context: {decode(context[0].tolist())}')
        sample = self.generate(context, new_tokens)
        text = decode(sample[0].tolist())
        print(f'Sample: {text}')


model = BigramLanguageModel(vocab_size)
logits, loss = model(xb, yb)

model.sample()

# typical lr setting is 3e-4, but for small models we can use a much higher lr
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# training loop
batch_size = 32  # ??
num_steps = 10000
for step in range(num_steps):
    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if step == 0 or step == num_steps - 1:
        print(loss.item())

model.sample()
