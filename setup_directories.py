import os

# Create project directories
directories = [
    'dataset',
    'dataset/train',
    'dataset/test',
    'models',
    'results',
    'notebooks',
    'src',
    'utils'
]

for dir_path in directories:
    os.makedirs(dir_path, exist_ok=True)
    print(f"Created directory: {dir_path}")

print("Project structure setup complete!")
