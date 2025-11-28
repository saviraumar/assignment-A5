"""
Q3 – GRU Implementation, Training, Evaluation, and Text Generation

This script covers:
(a) Implement GRU from scratch (NO nn.GRU/nn.GRUCell)
(b) Train GRU on Einstein text dataset
(c) Evaluate next-character prediction accuracy
(d) Generate 250 characters of text from a prompt
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ======================================================
# ------------  Utility: Dataset Handling  -------------
# ======================================================

def build_vocab(text):
    """Build character-level vocab from text."""
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos


def encode_text(text, stoi):
    """Encode full text to indices."""
    return torch.tensor([stoi[ch] for ch in text], dtype=torch.long)


def one_hot(indices, vocab_size):
    """Convert indices (batch,) to one-hot (batch, vocab_size)."""
    out = torch.zeros(indices.size(0), vocab_size)
    out[torch.arange(indices.size(0)), indices] = 1.0
    return out


def make_sequences(encoded, seq_len):
    """
    Turn [x0, x1, x2, ..., xN] into many (sequence, next_char) pairs.
    Each input: length seq_len, target: next char index.
    """
    inputs = []
    targets = []
    for i in range(len(encoded) - seq_len):
        seq = encoded[i:i + seq_len]
        target = encoded[i + seq_len]
        inputs.append(seq)
        targets.append(target)
    return torch.stack(inputs), torch.stack(targets)

def make_dataloader(text, seq_len=40, batch_size=64):
    stoi, itos = build_vocab(text)
    vocab_size = len(stoi)

    encoded = encode_text(text, stoi)  # (N,)

    X_idx, Y = make_sequences(encoded, seq_len)  # X_idx: (num_seq, seq_len), Y: (num_seq,)

    num_seq = X_idx.size(0)

    # One-hot encode: first flatten, then reshape back
    X_one_hot = one_hot(X_idx.view(-1), vocab_size)            # (num_seq*seq_len, vocab_size)
    X_one_hot = X_one_hot.view(num_seq, seq_len, vocab_size)   # (num_seq, seq_len, vocab_size)

    # IMPORTANT: here first dim = num_seq, same as Y
    dataset = TensorDataset(X_one_hot, Y)                      # OK now
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return loader, stoi, itos, vocab_size


# ======================================================
# ------------  (a) GRU Implementation  ----------------
# ======================================================

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """
        input_size: vocab_size (one-hot)
        hidden_size: number of GRU units
        output_size: vocab_size (for next-char prediction)
        """
        super().__init__()
        self.hidden_size = hidden_size

        # Update gate z_t
        self.x2z = nn.Linear(input_size, hidden_size, bias=True)
        self.h2z = nn.Linear(hidden_size, hidden_size, bias=False)

        # Reset gate r_t
        self.x2r = nn.Linear(input_size, hidden_size, bias=True)
        self.h2r = nn.Linear(hidden_size, hidden_size, bias=False)

        # Candidate hidden state ĥ_t
        self.x2h = nn.Linear(input_size, hidden_size, bias=True)
        self.h2h = nn.Linear(hidden_size, hidden_size, bias=False)

        # Output layer: h_t → logits
        self.out = nn.Linear(hidden_size, output_size)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def step(self, x_t, h_prev):
        """
        One GRU time step.
        x_t:    (batch, input_size)
        h_prev: (batch, hidden_size)
        returns: h_t, y_t (y_t = softmax over output_size)
        """
        # Update gate
        z_t = self.sigmoid(self.x2z(x_t) + self.h2z(h_prev))
        # Reset gate
        r_t = self.sigmoid(self.x2r(x_t) + self.h2r(h_prev))
        # Candidate hidden
        h_hat = self.tanh(self.x2h(x_t) + self.h2h(r_t * h_prev))
        # New hidden state
        h_t = (1 - z_t) * h_prev + z_t * h_hat
        # Output
        logits = self.out(h_t)
        y_t = F.softmax(logits, dim=-1)
        return h_t, y_t

    def forward(self, x_seq):
        """
        x_seq: (seq_len, batch, input_size)
        returns: outputs: (seq_len, batch, output_size)
        """
        seq_len, batch_size, _ = x_seq.shape
        h_t = torch.zeros(batch_size, self.hidden_size, device=x_seq.device)
        outputs = []
        for t in range(seq_len):
            x_t = x_seq[t]  # (batch, input_size)
            h_t, y_t = self.step(x_t, h_t)
            outputs.append(y_t)
        return torch.stack(outputs, dim=0)

    def predict(self, x_init, n_steps, idx_to_one_hot):
        """
        Autoregressive prediction:
        x_init: (seq_len, 1, input_size) — initial one-hot sequence
        n_steps: number of characters to generate
        idx_to_one_hot: function idx -> (1, input_size) tensor
        returns: list of predicted indices
        """
        seq_len, batch_size, _ = x_init.shape
        assert batch_size == 1

        # Run through initial sequence to get final hidden state
        h_t = torch.zeros(1, self.hidden_size, device=x_init.device)
        for t in range(seq_len):
            x_t = x_init[t]  # (1, input_size)
            h_t, y_t = self.step(x_t, h_t)

        preds = []
        x_t = x_init[-1]  # last input

        for _ in range(n_steps):
            h_t, y_t = self.step(x_t, h_t)
            pred_idx = torch.argmax(y_t, dim=-1).item()
            preds.append(pred_idx)
            x_t = idx_to_one_hot(pred_idx).to(x_init.device)  # (1, input_size)

        return preds


# ======================================================
# ------------  (b) Training the GRU  ------------------
# ======================================================

def train_gru(model, loader, num_epochs=10, lr=1e-3, device="cpu"):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for x_batch, y_batch in loader:
            # x_batch: (batch, seq_len, vocab_size) from DataLoader
            # We need: (seq_len, batch, vocab_size) for GRU
            x_batch = x_batch.permute(1, 0, 2).to(device)  # (seq_len, batch, input_size)
            y_batch = y_batch.to(device)                   # (batch,)

            optimizer.zero_grad()
            outputs = model(x_batch)                       # (seq_len, batch, vocab_size)
            final_outputs = outputs[-1]                    # (batch, vocab_size)

            loss = criterion(final_outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss = {avg_loss:.4f}")


# ======================================================
# ------------  (c) Accuracy Evaluation  ---------------
# ======================================================

def evaluate_accuracy(model, loader, device="cpu"):
    model.to(device)
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.permute(1, 0, 2).to(device)  # (seq_len, batch, vocab_size)
            y_batch = y_batch.to(device)

            outputs = model(x_batch)
            final_outputs = outputs[-1]                    # (batch, vocab_size)
            preds = torch.argmax(final_outputs, dim=-1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

    acc = correct / total
    print(f"\nNext-character accuracy: {acc * 100:.2f}%")
    return acc


# ======================================================
# ------------  (d) Text Generation  -------------------
# ======================================================

def idx_to_one_hot_fn(idx, vocab_size):
    """Return one-hot (1, vocab_size) tensor for index idx."""
    v = torch.zeros(1, vocab_size)
    v[0, idx] = 1.0
    return v


def decode_indices(indices, itos):
    """Turn list of indices back to a string."""
    return "".join(itos[i] for i in indices)


def generate_text(model, itos, stoi, prompt, seq_len, vocab_size, length=250, device="cpu"):
    model.to(device)
    model.eval()

    #  Remove characters that do not exist in vocabulary
    filtered_prompt = ''.join(ch for ch in prompt if ch in stoi)

    if len(filtered_prompt) == 0:
        raise ValueError("\n Prompt contains no known characters.\n"
                         "Try a prompt from the training text like 'relativity' or 'einstein'.\n")

    # Encode
    prompt_idx = torch.tensor([stoi[ch] for ch in filtered_prompt], dtype=torch.long)

    # One-hot helper
    def one_hot_idx(idx):
        v = torch.zeros(1, vocab_size)
        v[0, idx] = 1.0
        return v

    # Convert prompt to (seq_len,1,vocab)
    x_oh = torch.stack([one_hot_idx(i)[0] for i in prompt_idx], dim=0)
    x_oh = x_oh.unsqueeze(1).to(device)

    # Generate characters
    preds = model.predict(
        x_init=x_oh,
        n_steps=length,
        idx_to_one_hot=lambda idx: one_hot_idx(idx).to(device)
    )

    output = filtered_prompt + ''.join(itos[i] for i in preds)

    print("\n Generated Text Output \n")
    print(output)
    print("\n-------------------------------------------")

    return output

# ======================================================
# ----------------------- MAIN -------------------------
# ======================================================

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 🔹 Load Einstein text (you can replace path or paste text directly)
    try:
        with open("einstein.txt", "r", encoding="utf-8") as f:
            raw_text = f.read()
    except FileNotFoundError:
        print("Could not find 'einstein.txt'. Please place the Einstein text file in this folder.")
        raw_text = "relativity is a theory by albert einstein." * 200  # fallback dummy text

    # Create DataLoader
    seq_len = 40
    batch_size = 64
    loader, stoi, itos, vocab_size = make_dataloader(raw_text, seq_len=seq_len, batch_size=batch_size)

    # Instantiate model
    hidden_size = 128
    model = GRU(input_size=vocab_size, hidden_size=hidden_size, output_size=vocab_size)

    # ------- (b) Train -------
    print("\n--- Training GRU ---")
    train_gru(model, loader, num_epochs=15, lr=1e-3, device=device)

    # ------- (c) Evaluate -------
    print("\n--- Evaluating Accuracy ---")
    acc = evaluate_accuracy(model, loader, device=device)

    # ------- (d) Generate Text -------
    print("\n--- Generating Text ---")
    prompt = "relativity"  # SAFE & DEFINITELY IN DATASET
    generate_text(model, itos, stoi, prompt, seq_len, vocab_size, length=250, device=device)
