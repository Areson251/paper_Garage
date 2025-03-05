import os
import re

def rename_files(directory):
    if not os.path.isdir(directory):
        print("Указанная папка не найдена.")
        return
    
    for filename in os.listdir(directory):
        old_path = os.path.join(directory, filename)
        
        if not os.path.isfile(old_path):
            continue
        
        new_filename = re.sub(r'_sample_\d+_\d+', '', filename)
        new_path = os.path.join(directory, new_filename)
        
        if old_path != new_path:
            os.rename(old_path, new_path)
            print(f'Renamed: "{filename}" → "{new_filename}"')
    
    print("Готово!")

folder_path = "datasets/orig_test"
rename_files(folder_path)
