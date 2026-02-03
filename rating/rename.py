import os

# Directory where the screenshots are located
directory = 'screenshots'

# Loop through the range of files from screenshot_0.png to screenshot_99.png
for i in range(100):
    # Create old and new filenames
    old_name = f"screenshot_{i}.png"
    new_name = f"screenshot_{i + 100}.png"
    
    # Define the full file paths
    old_path = os.path.join(directory, old_name)
    new_path = os.path.join(directory, new_name)
    
    # Check if the file exists and rename it
    if os.path.exists(old_path):
        os.rename(old_path, new_path)
        print(f"Renamed: {old_name} -> {new_name}")
    else:
        print(f"File {old_name} not found.")
