import os
import glob
from pathlib import Path
import re
import shutil
from  contextlib import contextmanager,redirect_stderr,redirect_stdout

"""
Basic io functions.
"""

# TODO
# - move hsfm.utils.run_command here 

@contextmanager
def redirect_stdout_stderr(stdout_fn=None, 
                           stderr_fn=None):
    '''
    Writes stdout and/or stderr to file.
    Use os.devnull as the file name to silence entirely.
    '''
    Path(stdout_fn).parent.mkdir(parents=True, exist_ok=True)
    Path(stderr_fn).parent.mkdir(parents=True, exist_ok=True)
    
    if stdout_fn and stderr_fn:
        with open(stdout_fn, 'w') as stdout, open(stderr_fn, 'w') as stderr:
            with redirect_stdout(stdout) as out, redirect_stderr(stderr) as err:
                yield (err, out)
    elif stdout_fn:
        with open(stdout_fn, 'w') as stdout:
            with redirect_stdout(stdout) as out:
                yield out
    elif stderr_fn:
        with open(stderr_fn, 'w') as stderr:
            with redirect_stdout(stderr) as err:
                yield err

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
    
    
def rename_file(source_file_name, 
                pattern=None,
                new_pattern=None,
                destination_file_path=None,
                destination_file_extension=None,
                write=False):
    
    file_path, file_name, file_extention = split_file(source_file_name)
    
    if not isinstance(pattern, type(None)):
        
        if new_pattern==None:
            new_pattern=''
        
        file_name = file_name.replace(pattern, new_pattern)
        
    if destination_file_path:
        create_dir(destination_file_path)
        file_path = destination_file_path
        
    if destination_file_extension:
        file_extention = destination_file_extension
            
    destination_file_name = os.path.join(file_path, file_name+file_extention)
    
    if write==True:
        shutil.copy2(source_file_name,destination_file_name)
    else:
        return destination_file_name

def batch_rename_files(source_file_path,
                       file_extension=None,
                       unique_id_pattern=None,
                       pattern=None,
                       new_pattern=None,
                       destination_file_path=None,
                       destination_file_extension=None):
    
    create_dir(destination_file_path)
    
    var_list = [file_extension, 
                unique_id_pattern, 
                pattern, 
                new_pattern, 
                destination_file_path, 
                destination_file_extension]
    
    if all(isinstance(x, type(None)) for x in var_list):
        print("No options provided. Source and destination files identical.")
    
    else:
        if file_extension:
            files = glob.glob(os.path.join(source_file_path,'**','*'+ file_extension), recursive=True)
        else:
            files = glob.glob(os.path.join(source_file_path,'**','*'), recursive=True)

        if unique_id_pattern:
            files = [x for x in files if unique_id_pattern in x] 
        
        for source_file_name in files:
            new_file = rename_file(source_file_name, 
                                   pattern=pattern,
                                   new_pattern=new_pattern,
                                   destination_file_path=destination_file_path,
                                   destination_file_extension=destination_file_extension)

            if source_file_name != new_file:
                shutil.copy2(source_file_name,new_file)

def retrieve_match(pattern, file_list):
    for i in file_list:
        if pattern in i:
            return i
def find_number_in_string(string):
    numbers = [float(n) for n in re.findall(r"[-+]?\d*\.?\d+|[-+]?\d+",string)]
    return numbers