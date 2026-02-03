# Generated from /Users/cui/Documents/GitHub/HighDimSpatialStatistics/temporary_code/rename_some_files.ipynb on 2026-02-03 10:30:29
# DO NOT EDIT: regenerate via scripts/tools/convert_notebooks.py

# %%
import os
import glob

# Define the directory containing the .sh files
directory = os.path.expanduser("~/project/")

# Get all .sh files in the directory
sh_files = glob.glob(os.path.join(directory, "*.sh"))

# Loop through each .sh file
for filepath in sh_files:
    # Open the file for reading
    with open(filepath, 'r') as file:
        content = file.read()
    
    # Replace "--mail-type=ALL" with "--mail-type=END, FAIL"
    updated_content = content.replace("--mail-type=ALL", "--mail-type=END,FAIL")
    
    # Write the updated content back to the file
    with open(filepath, 'w') as file:
        file.write(updated_content)

print("Replacement completed.")

