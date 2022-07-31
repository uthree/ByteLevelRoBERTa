from model import *
from dataset import *

import torch
import torch.nn as nn
import torch.optim as optim

import os
from tqdm import tqdm

BATCH_SIZE = 16
MAX_LEN = 128
NUM_EPOCH = 500

if os.path.exists("./model.pt"):
    print("Loaded model.")
    model = torch.load("./model.pt")
else:
    print("Initializing model...")
    model = BERT(
            ByteEmbedding(embedding_dim=256),
            nn.TransformerEncoder(nn.TransformerEncoderLayer(256, 4, activation='gelu', batch_first=True), 6),
            ByteUnembedding(embedding_dim=256),
            PositionalEmbedding(d_model=256, max_length=MAX_LEN)
            )


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = Tokenizer()

dataset = TextDataset(["/home/uthree/ao-childes-torch-dataset/aochildes.txt"])


optimizer = optim.RAdam(model.parameters(), lr=1e-4)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
criterion = nn.CrossEntropyLoss()

model.to(device)
for epoch in range(NUM_EPOCH):
    bar_batch = tqdm(total = len(dataset))
    for i, data in enumerate(dataloader):
        optimizer.zero_grad()
        src = torch.LongTensor(tokenizer.tokenize_sentences(data, max_length=MAX_LEN, mask_prob=0.15)).to(device)
        tgt = torch.LongTensor(tokenizer.tokenize_sentences(data, max_length=MAX_LEN)).to(device)

        out = model(src)
        loss = criterion(torch.flatten(out, 0, 1), torch.flatten(tgt, 0, 1))
        
        loss.backward()
        optimizer.step()
        bar_batch.set_description(desc=f"loss: {loss.item():.4f}")
        tqdm.write(tokenizer.untokenize_single(list(src[0].cpu().numpy()))+ "\n" + tokenizer.untokenize_single(list(torch.argmax(out[0], dim=1).cpu().numpy())))
        bar_batch.update(len(data))
        if i % 1000 == 0:
            torch.save(model, "./model.pt")
