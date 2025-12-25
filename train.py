import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, AdamW

# 1. Configuration for CPU Training
device = torch.device("cpu")
model_path = "./trocr_small_local" # Your local folder
batch_size = 2  # Keep low for CPU RAM

# 2. Custom Dataset Class
class OCRDataset(Dataset):
    def __init__(self, df, root_dir, processor, max_target_length=128):
        self.df = df
        self.root_dir = root_dir
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.df['file_name'][idx]
        text = self.df['text'][idx]
        image = Image.open(self.root_dir + file_name).convert("RGB")
        
        # Process image and text
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        labels = self.processor.tokenizer(text, padding="max_length", 
                                          max_length=self.max_target_length).input_ids
        
        # Important: set pad tokens to -100 so they are ignored by loss
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

        return {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}

# 3. Load Model and Processor
processor = TrOCRProcessor.from_pretrained(model_path)
model = VisionEncoderDecoderModel.from_pretrained(model_path).to(device)

# 4. Prepare Data
train_df = pd.read_csv("metadata.csv")
train_dataset = OCRDataset(df=train_df, root_dir="./images/", processor=processor)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 5. Simple Training Loop
optimizer = AdamW(model.parameters(), lr=5e-5)
model.train()

for epoch in range(3):  # Start with 3 epochs for testing
    for batch in train_dataloader:
        inputs = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(pixel_values=inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"Loss: {loss.item()}")

# 6. Save your fine-tuned model
model.save_pretrained("./finetuned_trocr_small")
processor.save_pretrained("./finetuned_trocr_small")