import os
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

def generate_captions_for_images_in_folder(folder_path, output_file='captions.txt'):
    # Check if CUDA is available and set device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the pre-trained BLIP model and processor
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    model.to(device)

    # Open the output file
    with open(output_file, 'w') as f:
        # Iterate over all files in the folder
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                # Construct the full file path
                file_path = os.path.join(folder_path, filename)
                
                # Open the image
                image = Image.open(file_path).convert('RGB')
                
                # Process the image and generate the caption
                inputs = processor(images=image, return_tensors="pt").to(device)
                outputs = model.generate(**inputs)
                caption = processor.decode(outputs[0], skip_special_tokens=True)
                
                # Write the filename and caption to the output file
                f.write(f"{filename}: {caption}\n")
                print(f"Generated caption for {filename}: {caption}")


if __name__ == "__main__":
    folder_path = 'example_batch'  # Change this to your folder path
    generate_captions_for_images_in_folder(folder_path, output_file=f'{folder_path}_blip_captions.txt')
