import os

# Define the folder containing the images
folder_path = 'example_batch'

# Define the captions
captions = {
    "abstract_best_8k_cogvlm_composition_elaine-de-kooning_untitled-bull-1973.png": "a painting of a bull with a colorful background",
    "abstract_best_8k_cogvlm_eye_movement_2_richard-pousette-dart_opaque-harmony-1943.png": "a painting of a man with a hat",
    "abstract_best_8k_cogvlm_eye_movement_2_sam-francis_all-red-1964.png": "a painting with many colors and spots",
    "abstract_best_8k_cogvlm_movement_sam-gilliam_stand-1973.png": "a painting of a sun with a blue sky in the background",
    "abstract_deepseek_15k_llava_movement_friedel-dzubas_red-flight-1957.png": "a painting of a man riding a horse",
    "abstract_deepseek_15k_llava_proportion_bradley-walker-tomlin_number-13-1949.png": "a painting of a man sitting on a dock",
    "abstract_deepseek_15k_llava_symmetry_asymmetry_1_jackson-pollock_moon-woman-1942(1).png": "two men by paul kreis",
    "abstract_deepseek_25k_cogvlm_balance_elements_alice-baber_wheel-of-day-1971.png": "a painting of a flower with green leaves",
    "abstract_deepseek_25k_cogvlm_foreground_background_4_vasile-dobrian_the-fish.png": "a fish swimming in the ocean with rocks"
}

# Create the HTML content
html_content = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Image Captions</title>
    <style>
        body {
            font-family: Arial, sans-serif;
        }
        .container {
            display: grid;
            grid-template-columns: repeat(3, 310px);
            gap: 5px;
        }
        .item {
            display: flex;
            align-items: center;
            border: 1px solid #ccc;
            padding-left: 10px;
            max-width: 250px;
        }
        .item img {
            max-width: 100px;
            margin-left: 10px;
        }
        .caption {
            font-size: 14px;
        }
    </style>
</head>
<body>
    <h1>Image Captions</h1>
    <div class="container">
'''

# Add images and captions to the HTML content
for filename, caption in captions.items():
    file_path = os.path.join(folder_path, filename)
    html_content += f'''
        <div class="item">
            <div class="caption">{caption}</div>
            <img src="{file_path}" alt="{caption}">
        </div>
    '''

# Close the HTML tags
html_content += '''
    </div>
</body>
</html>
'''

# Write the HTML content to a file
with open('blip_captions_interface.html', 'w') as f:
    f.write(html_content)

print("HTML file 'blip_captions_interface.html' has been created.")

