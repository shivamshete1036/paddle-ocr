import os

# Define the folder and file path
folder_path = "./ppocr_utils"
file_path = os.path.join(folder_path, "en_dict.txt")

# Create the folder if it doesn't exist
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# Standard 96 characters for English PaddleOCR
# This includes numbers, lowercase, uppercase, and common symbols
characters = (
    "0123456789"
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ "
)

# Write each character to a new line
with open(file_path, "w", encoding="utf-8") as f:
    for char in characters:
        f.write(char + "\n")

print(f"Success! Dictionary created at: {file_path}")
print(f"Total characters: {len(characters)}")