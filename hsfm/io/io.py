import os

def create_dir(directory):
    if directory == None:
        return None
    
    else:
        if not os.path.exists(directory):
            os.makedirs(directory)
        return os.path.abspath(directory)


def split_file(file_path_and_name):
    file_path = os.path.split(file_path_and_name)[0]
    file_name = os.path.splitext(os.path.split(file_path_and_name)[-1])[0]
    file_extension = os.path.splitext(os.path.split(file_path_and_name)[-1])[-1]
    
    return file_path, file_name, file_extension
    
    
def replace_string_in_file(input_file, output_file, string, new_string):
    
    file_in = open(input_file).read()
    file_in = file_in.replace(string, new_string)
    
    file_out = open(output_file, 'w')
    file_out.write(file_in)
    
    file_out.close()