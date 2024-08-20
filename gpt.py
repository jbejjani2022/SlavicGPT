import os
import time
import logging
import requests
from contextlib import nullcontext
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler as DS
from torch.nn.parallel import DistributedDataParallel as DDP


# set hyperparameters
split = 0.8  # the percentage of the dataset to be used for training - rest is used for valdiation
batch_size = 64  # the number of independent sequences that we will process in parallel
block_size = 256  # maximum context length for predictions
learning_rate = 3e-4
max_iters = 5000  # number of training steps
eval_interval = 500  # how often to evaluate the loss
eval_iters = 200  # number of batches to be evaluated during loss estimation
save_interval = 1000  # how often to save a model checkpoint
n_embd = 384  # number of embedding dimensions
n_heads = 6  # number of self-attention heads per transformer block
n_blocks = 6  # number of transformer blocks
dropout = 0.2  # dropout probability
out = 'gpt.log' # output log filename
wandb_log = True  # log to wandb
wandb_project = 'gpt'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
config = {k: globals()[k] for k in config_keys} # will be useful for wandb logging
checkpoint_dir = "./checkpoints" # path to model checkpoints
dataset = 'tinyshakespeare' # 'tinyrussianlit'

# Configure logging to an output file
logging.basicConfig(filename=out, 
                    filemode='a',  # appends to previous logs
                    level=logging.INFO,  # The logging level
                    format='%(message)s')  # Format of each log entry

# link to dataset
data_url = None
if dataset == 'tinyshakespeare':
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
elif dataset == 'tinyrussianlit':
    data_url = 'https://raw.githubusercontent.com/jbejjani2022/SlavicGPT/main/data/tiny-russian-lit/very_clean_tiny_russian_lit.txt'

data_file = f"input/{dataset}.txt"
data_file_path = os.path.join(os.path.dirname(__file__), data_file)
# Download data if not found in path
if not os.path.exists(data_file_path):
    logging.info(f'Downloading {dataset} dataset to {data_file_path}')
    if not data_url:
        raise ValueError(f"No URL given for dataset {dataset}")
    with open(data_file_path, 'w', encoding='utf-8') as f:
        f.write(requests.get(data_url).text)

ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?

if ddp:
    dist.init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging and checkpointing
    seed_offset = ddp_rank # each process gets a different seed
else:
    # if not ddp, we are running on one gpu, one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
torch.manual_seed(3 + seed_offset)    

if master_process:
    logging.info(f'DDP run? {ddp}')
    logging.info(f'world size = {ddp_world_size}')
    logging.info(f'Using {dtype} data type')
    tokens_per_iter = ddp_world_size * batch_size * block_size
    logging.info(f"There will be {tokens_per_iter:,} tokens per iteration, for {max_iters} iterations")
    
logging.info(f"{device} initialized")

device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
# initialize context for automatic mixed precision operations if training on gpu
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Read in data
with open(data_file_path, 'r', encoding='utf-8') as f:
    text = f.read()
    
# find the unique characters that occur in the text
chars = sorted(list(set(text)))
vocab = ''.join(chars)
vocab_size = len(chars)

if master_process:
    logging.info(f'{len(text) / 1e6} million characters loaded from the {dataset} dataset.')
    logging.info(f'Dataset vocabulary: {vocab}')
    logging.info(f'Vocabulary size: {vocab_size}')

# create a simple character-level tokenizer:
# a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


# encoder: convert string to list of integers
def encode(s): return [stoi[c] for c in s]
# decoder: convert list of integers to string
def decode(l): return ''.join([itos[i] for i in l])


# split data into train and validation sets
data = torch.tensor(encode(text), dtype=torch.long)
if master_process:
    logging.info(f'Num tokens in dataset: {len(data) / 1e6} million')  # for a character-level language model, num characters = num tokens
n = int(split*len(data))
train_data = data[:n]
val_data = data[n:]


# data loading
def get_batch(split):
    """Generate a batch of data consisting of inputs x and targets y."""
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    if device_type == 'cuda':
        # pin arrays x, y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()  # avoids unnecessarily allocating memory for storing gradients; we will not backprop. losses computed during evaluation, so we don't need PyTorch to track operations
def estimate_loss():
    """Evaluate the model on the train and val sets.
    Estimates the loss because we only process `eval_iters`
    random batches from each of the train and val sets.
    """
    out = {}
    model.eval()
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
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B, T, C)
        q = self.query(x)  # (B, T, C)
        # compute attention scores i.e. "affinities" between each query and key
        # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
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
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Run the heads in parallel and concatenate the results over the channel dimension
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """A linear layer followed by a non-linearity."""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
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
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa_heads(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class GPT(nn.Module):
    """A decoder-only transformer for generating text, as done in OpenAI's GPT."""

    def __init__(self):
        super().__init__()
        # each token reads off the logits (input to softmax) for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # each token position/timestep in a given block gets its own embedding vector
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # transformer blocks
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_heads) for _ in range(n_blocks)])
        self.ln_f = nn.LayerNorm(n_embd)  # final layer norm
        # language modeling head: maps final token embeddings to logits
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
        x = self.ln_f(x)  # (B, T, C)
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
        logging.info(f'Context: {decode(context[0].tolist())}')
        sample = self.generate(context, new_tokens)
        text = decode(sample[0].tolist())
        return text
        
    def num_params(self):
        # Calculate total number of parameters in the model
        return sum(p.numel() for p in self.parameters())


model = GPT().to(device)

if master_process:
    logging.info(f'Num parameters in model: {model.num_params() / 1e6} million')
    # generate text from untrained model
    logging.info('\nSample text generated by untrained model \n' + '-' * 50)
    logging.info(f'Sample:{model.sample_text()}')
    logging.info('-' * 50 + '\n')

# Wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

raw_model = model.module if ddp else model # unwrap DDP container if needed

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
# Initialize GradScaler. If enabled=False, scaler is a no-op
enabled = dtype == 'float16'
scaler = torch.cuda.amp.GradScaler(enabled=enabled)
if master_process:
    logging.info(f'GradScaler enabled? {enabled}')


def save_checkpoint(i):
    """Save model checkpoint at training step i."""
    model_path = f"{checkpoint_dir}/step_{i}.pt"
    os.makedirs(os.path.dirname(model_path), exist_ok=True) 
    torch.save({
                'step': i,
                'model_state_dict': raw_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                }, model_path)
    logging.info(f'Saved model checkpoint to {model_path}')
    
    
def load_latest_checkpoint():
    """Load the latest model checkpoint found in the checkpoints dir."""
    if not os.path.exists(checkpoint_dir):
        logging.error(f"Checkpoint directory '{checkpoint_dir}' does not exist.")
        return
    # Get all the checkpoint files
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")]
    if not checkpoints:
        logging.warning("No checkpoints available.")
        return
    # Sort the checkpoints based on the training step number in their filenames
    # Each checkpoint is 'step_i.pt', where i is the training step number
    checkpoints.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    # Get the path of the latest checkpoint
    latest_checkpoint = os.path.join(checkpoint_dir, checkpoints[-1])
    # Load the checkpoint
    logging.info(f"Loading the latest checkpoint, found at {latest_checkpoint}")
    checkpoint = torch.load(latest_checkpoint)
    # Load the model and optimizer states
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # so we can resume training from the checkpoint


# Uncomment to resume training from the latest checkpoint
# load_latest_checkpoint()

# wandb logging
if wandb_log and master_process:
    import wandb
    logging.info(f'Initializing wandb...')
    wandb_run_name = os.getenv('WANDB_RUN_NAME', wandb.util.generate_id())
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)
    
    
# TRAINING LOOP
X, Y = get_batch('train') # fetch the very first batch
if master_process:
    t0 = time.time()
for i in range(max_iters):
    # periodically get the loss on train and val sets
    if master_process and (i == 0 or (i + 1) % eval_interval == 0):
        losses = estimate_loss()
        if i == 0:
            logging.info(f"losses before training: train loss = {losses['train']:.4f}, val loss = {losses['val']:.4f}")
        else:
            logging.info(f"step {i + 1}/{max_iters}: train loss = {losses['train']:.4f}, val loss = {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": i,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
            })

    # periodically save model checkpoint
    if master_process and (i + 1) % save_interval == 0:
        save_checkpoint(i)
    
    with ctx:
        logits, loss = model(X, Y)
    
    # immediately async prefetch next batch while model is doing the forward pass on the GPU
    X, Y = get_batch('train')
    # backward pass, with gradient scaling if training in fp16
    scaler.scale(loss).backward()
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients
    optimizer.zero_grad(set_to_none=True)

if master_process:
    t1 = time.time()
    dt = t1 - t0
    logging.info(f'Training time: {dt / 60:.2f} min')
    # Generate text from the trained model
    logging.info('\nSample text generated by trained model \n' + '-' * 50)
    logging.info(f'Sample:{raw_model.sample_text()}')
    logging.info('-' * 50 + '\n')

# Destroy processes
if wandb_log and master_process:
    wandb.finish()
if ddp:
    dist.destroy_process_group()
