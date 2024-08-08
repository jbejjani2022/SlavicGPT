import torch
import torch.nn as nn
from torch.nn import functional as F

# set random seed for reproducibility
torch.manual_seed(3)

# set hyperparameters
split = 0.8  # the percentage of the dataset to be used for training - remaining is used for valdiation
batch_size = 32  # the number of independent sequences that we will process in parallel
block_size = 8  # maximum context length for predictions
learning_rate = 1e-3
max_iters = 5000  # number of training steps
eval_interval = 500  # how often to evaluate the loss
eval_iters = 200  # number of batches to be evaluated during loss estimation
n_embd = 32  # number of embedding dimensions
n_heads = 4  # number of self-attention heads per transformer block
n_blocks = 3  # number of transformer blocks
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# path to data file
data_path = 'data/tiny-russian-lit/very_clean_tiny_russian_lit.txt'

# read in data
with open(data_path, 'r', encoding='utf-8') as f:
    text = f.read()

# find the unique characters that occur in the text
chars = sorted(list(set(text)))
vocab = ''.join(chars)
vocab_size = len(chars)

# create a simple character-level tokenizer:
# a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


# encoder: convert string to list of integers
def encode(s): return [stoi[c] for c in s]
# decoder: convert list of integers to string
def decode(l): return ''.join([itos[i] for i in l])


# split data into train and validation sets to test for overfitting
data = torch.tensor(encode(text), dtype=torch.long)
n = int(split*len(data))
train_data = data[:n]
val_data = data[n:]


# data loading
def get_batch(split):
    """Generate a batch of data consisting of inputs x and targets y."""
    data = train_data if split == 'train' else val_data
    # generate `batch_size` random offsets in the interval [0, len(data) - batch_size)
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()  # avoids unnecessarily allocating memory for storing gradients; we will not backprop. losses computed during evaluation, so we don't need PyTorch to track operations
def estimate_loss():
    """Evaluate the model on the train and val sets.
    Estimates the loss because we only evaluate `eval_iters`
    random batches from each of the train and val sets.
    """
    out = {}
    model.eval()  # doesn't actually do anything for the bigram, which behaves the same in evaluation and training mode since there are no dropout and batchnorm layers - but will be necessary for transformer
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    """A single head of self-attention."""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(
            torch.ones(block_size, block_size)))  # a buffer is not a parameter of the model

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B, T, C)
        q = self.query(x)  # (B, T, C)
        # compute attention scores i.e. "affinities" between each query and key
        # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        # perform weighted aggregation of values
        v = self.value(x)  # (B, T, C)
        out = wei @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out


class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel.
    Think of these heads as multiple independent channels of 
    communication between tokens."""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

    def forward(self, x):
        # Run the heads in parallel and concatenate the results over the channel dimension
        return torch.cat([h(x) for h in self.heads], dim=-1)


class FeedForward(nn.Module):
    """A linear layer followed by a non-linearity."""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication followed by computation."""

    def __init__(self, n_embd, n_heads):
        # n_embd: embedding dimension, n_heads: the number of heads of self-attention
        super().__init__()
        head_size = n_embd // n_heads
        # self-attention multi-head: 'communication'
        self.sa_heads = MultiHeadAttention(n_heads, head_size)
        # feedforward: 'computation'
        self.ff = FeedForward(n_embd)

    def forward(self, x):
        x = x + self.sa_heads(x)
        x = x + self.ff(x)
        return x


class BigramLanguageModel(nn.Module):

    def __init__(self, h=n_blocks):
        # h: the number of transformor blocks
        super().__init__()
        # each token reads off the logits (input to softmax) for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # each token position/timestep in a given block gets its own embedding vector
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # transformer blocks
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_heads) for _ in range(h)])
        # language modeling head: maps token embeddings to logits
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B, T) tensor of integers (B = # batches, T = # timesteps/block size)
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)  # (B, T, n_embd)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=device))  # (T, n_embd)
        # (B, T, C) - PyTorch broadcasts the pos_emb across batch dim
        x = tok_emb + pos_emb
        x = self.blocks(x)  # (B, T, C)
        logits = self.lm_head(x)  # (B, T, vocab_size)

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
            # crop idx to the last block_size tokens, since pos_emb only has embeddings for the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get predictions
            logits, loss = self(idx_cond)  # calls the forward function
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
    def sample_text(self, context=torch.tensor([encode('\n')], device=device), new_tokens=250):
        print(f'Context: {decode(context[0].tolist())}')
        sample = self.generate(context, new_tokens)
        text = decode(sample[0].tolist())
        print(f'Sample: {text}\n')


model = BigramLanguageModel().to(device)
# generate text from untrained model
print(f'\nSample text generation from untrained bigram')
print('-' * 50)
model.sample_text()

# typical lr setting is 3e-4, but for small models we can use a much higher lr
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# training loop
for iter in range(max_iters):
    # periodically get the loss on train and val sets
    if (iter + 1) % eval_interval == 0:
        losses = estimate_loss()
        print(
            f"step {iter + 1}/{max_iters}: train loss = {losses['train']:.4f}, val loss = {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss and update parameters
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate text from the trained model
print(f'\nSample text generation from trained bigram')
print('-' * 50)
model.sample_text()
