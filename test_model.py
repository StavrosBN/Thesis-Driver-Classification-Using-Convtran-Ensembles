import torch
from models.model import ConvTran

# Same dimentions with the real data (windowing.py):
# X shape was (N, 60, 30) = (batch, seq_len=60, num_features=30)
# The model wants (batch, num_features, seq_len) -- so we transpose

batch_size = 4
num_features = 30
seq_len = 60
num_classes = 3  #ex. cleaned_1 (A, B, C)

# Dummy data, on the correct shape (batch, features, seq_len)
dummy_x = torch.randn(batch_size, num_features, seq_len)

model = ConvTran(
    num_features=num_features,
    seq_len=seq_len,
    num_classes=num_classes,
    emb_size=16,
    num_heads=8,
    dim_ff=256,
    dropout=0.01
)

# Test 1: classification
output = model(dummy_x)
print("Classification output shape:", output.shape)
# We expect: (4, 3)  -> 4 samples, 3 classes

# Test 2: embedding 
embedding = model(dummy_x, return_embedding=True)
print("Embedding shape:", embedding.shape)
# We expect: (4, 16)  -> 4 samples, 16 dimentions 

print("\nΑριθμός εκπαιδεύσιμων παραμέτρων:", sum(p.numel() for p in model.parameters() if p.requires_grad))