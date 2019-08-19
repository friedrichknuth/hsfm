import os

def split_file(file_path_and_name):
    file_path = os.path.split(file_path_and_name)[0]
    file_name = os.path.split(file_path_and_name)[-1].split('.')[0]
    file_extension = '.'+os.path.split(file_path_and_name)[-1].split('.')[-1]
    
    return file_path, file_name, file_extension