#
# package containing all os related functions
#
import os
import pandas as pd
from IPython.display import display
import platform

# use True/False instead of 1/0 !

#
def creation_date(path_to_file, last_mod=False):
    """
    Try to get the date that a file was created, falling back to when it was
    last modified if that isn't possible.
    See http://stackoverflow.com/a/39501288/1709587 for explanation.
    
    If last modification date should be returned, set <last_mod> = True
    [default (last_mod=False) returns timestamp of when file was created]
    
    To convert to human readable time formats use (e.g.)
    datetime.datetime.fromtimestamp(output)
    datetime.datetime.utcfromtimestamp(output).strftime("%Y-%m-%d")
    datetime.datetime.utcfromtimestamp(output).strftime("%H:%M:%S")
    """
    if platform.system() == 'Windows':
        if last_mod == True: # last modified
            return os.path.getmtime(path_to_file)
        elif last_mod == False: # created
            return os.path.getctime(path_to_file)
    else:
        stat = os.stat(path_to_file)
        try:
            return stat.st_birthtime
        except AttributeError:
            # We're probably on Linux. No easy way to get creation dates here,
            # so we'll settle for when its content was last modified.
            return stat.st_mtime

#
# print all subfolders in a parent folder
def return_all_folders(parent, full_path=1, print_list=1):
    """
    return all folders contained in the parent folder
    given to the function, i.e. <parent>.
    <full_path> =1 will return the full path, =0 only folders itself
    <print_list> =1 will also print, =0 will only return
    """
    # initialize
    subdirs = []
    # get all folder names within this folder
    for file_or_subdir in os.listdir(parent):
        temp_path = os.path.join(parent, file_or_subdir)
        # is temp_path a directory, then add to list
        if os.path.isdir(temp_path):
            if full_path == 1:
                subdirs.append((temp_path))
            elif full_path == 0:
                subdirs.append((file_or_subdir))
    # return (and print)
    if print_list == 1:
        for subdir in subdirs: print(subdir)
    #
    return subdirs

#
# print all subfolders in a parent folder
def return_all_files(parent, include_subs, file_type, print_list):
    """
    return all files of the type <file_type> contained in the parent
    folder given to the function, i.e. <folder>
    Pass file_type = '', if all file types should be returned
    If <include_subs> is 0, only files within the parent folder are
    printed and returned, if it is equal to 1 then all files are 
    returned, if equal to 2 then only files from subfolders are
    printed and returned 
    The files from subfolders are returned and printed as
    "sub folder \ file from sub folder"
    print_list =0 (return only), =1 (return and print list)
    """
    # get all files in parent
    if file_type == '': # all
        list_files = [x for x in os.listdir(parent)
                      if os.path.isfile(os.path.join(parent, x))]
    else: # only selected file type
        list_files = [x for x in os.listdir(parent)
                      if x.endswith(file_type)]
        
    # erase those files if only files in subfolders requested
    if include_subs == 2: list_files = []
    
    # all files in subfolders
    if include_subs >= 1:
        # get all folder and file names within parent folder
        for file_or_subdir in os.listdir(parent):
            # get all subfolders in parent folder
            sub_path = os.path.join(parent, file_or_subdir)
            # select the ones that are a subfolder
            if os.path.isdir(sub_path): 
                # extend all files list by files in sub
                if file_type == '': # all
                    sub_files = [os.path.join(file_or_subdir, x)
                                 for x in os.listdir(sub_path)]
                else: # only selected file type
                    sub_files = [os.path.join(file_or_subdir, x)
                                 for x in os.listdir(sub_path) if x.endswith(file_type)]
                # add
                list_files.extend(sub_files)
        
    # return list
    if print_list == 1:
        for file in list_files: print(file)
    return list_files

#
def read_file(path, file, print_info=1):
    """
    Read <path> + <file> (both str)
    
    If filetype .csv then load as pandas dataframe
    Return type is pd.DataFrame()
    
    If filetype is .txt then read file as string
    Return type is string
    
    If filetype is .dat then read as string interpreting everything
    before '\n\n\n' as the header (string) and everything after as 
    pandas dataframe
    Return type list with two elements, i.e. string and pd.DataFrame()
    
    If file doesn't exist a message is printed, and an empty object is
    returned. For filetypes not defined above an empty list is returned
    """
    # read as pandas dataframe
    path_full = os.path.join(path, file)
    if file[-4:] == '.csv': 
        if not os.path.isfile(path_full):
            if print_info >= 1:
                print('File does not exist:'); print(path_full)
            temp = pd.DataFrame()
        else:
            if print_info >= 2:
                print('Loaded file:'); print(path_full)
            temp = pd.read_csv(path_full)

    # read as string
    elif file[-4:] in ['.txt', '.mps']:
        if not os.path.isfile(path_full):
            if print_info >= 1:
                print('File does not exist:'); print(path_full)
            temp = ''
        else:
            if print_info >= 2:
                print('Loaded file:'); print(path_full)
            f = open(path_full, "r")
            temp = f.read()
    
    # read as header(string like .txt) + pandas dataframe (like .csv)
    elif file[-4:] == '.dat':
        if not os.path.isfile(path_full):
            if print_info >= 1:
                print('File does not exist:'); print(path_full)
            temp = ['', pd.DataFrame()]
        else: #$% FIX! df reading from string is a dummy line so far
            if print_info >= 2:
                print('Loaded file:'); print(path_full)
            f = open(path_full, "r")
            # split at triple new line
            # first part is header, second part is df
            str_split = f.read().split('\n\n\n')[0]
            temp = [str_split[0], pd.DataFrame(str_split[1])]

    else: # print info to add another "elif" in here
        print('filetype not supported for file:'); print(file)
        temp = []
        
    #
    return temp

#
def read_files(path, files, show_info=1):
    """    
    Return a list of ouput variables depending on file type for every
    file in <files> (list)
    <path> is base path that is combined with file elements in <files>
    (list)
    
    See <read_file> for details
    """
    return [read_file(path, file, print_info=show_info) for file in files]

#
def show_files(path, files, show_info=1):
    """
    Display full file contents for every file in <files> (list)
    [path +file]
    
    Doesn't return anything. See <read_file> for details
    """
    for file in files:
        print(os.path.join(path, file))
        temp = read_file(path, file, print_info=show_info)
        # csv: read as pandas dataframe
        if file[-4:] == '.csv': 
            if show_info >= 1: display(temp)
        # txt: read as string
        elif file[-4:] == '.txt': 
            if show_info >= 1: print(f'{temp}\n')
        # dat: header plus df
        elif file[-4:] == '.dat':
            print(temp[0]); display(temp[1])
        # print info to add another "elif" in here
        else: 
            print('filetype not supported for file:'); print(file)
    #
    return None

#
def find_file(path, file_type, search_str, show_info=0):
    """
    Find a file (of certain file_type starting with search_str) within
    a folder
    <folder> contains files to search from (str)
    <file_type> = 'txt', 'dat', 'bin', 'mps', etc. (str)
    <search_str> (str) is the beginning of the name of the file of 
    interest (str)
    
    Returns file path to file found or 
    warnings of no file or more than 1 file found
    """
    # (optionally) print overview
    if show_info >= 1:
        print(f'Searching file starting with {search_str} and ending in .{file_type}')
    
    # get all files in folder that match in terms of file type
    files = return_all_files(path, 0, file_type, 0)
    if show_info >= 2:
        print(f'Searching within all files of type {file_type}')
        print(files)
    
    # search file in all files
    count_match = 0 # initialize counter for number of matches (should be =1)
    for file in files:
        if search_str == file[0:len(search_str)]:
            if show_info >= 0.5: print(f'Found file: {file}')
            file_name = file
            count_match += 1
    
    # create file_path output or return empty string and warning/error message
    if count_match == 0: 
        print(f'WARNING. File starting with {search_str} and ending in .{file_type} not found!')
        file_name = ''
    elif count_match >= 2: 
        print(f'WARNING. More than one file found starting with {search_str} and ending in .{file_type}!')
        file_name = '' # overwrite to force error
    #
    return file_name

#
def find_files(path, file_type, search_str, show_info=0):
    """
    Find all files (of certain file_type starting with search_str) within a folder.
    <folder> contains files to search from (str).
    <file_type> = 'txt', 'dat', 'bin', 'mps', etc. (str)
    <search_str> is the beginning of the name of the file of interest (str).
    Returns file paths to files found or warnings if no file is found
    """
    # (optionally) print overview
    if show_info >= 1:
        print(f'Searching file starting with {search_str} and ending in .{file_type}')
    
    # get all files in folder that match in terms of file type
    files = return_all_files(path, 0, file_type, 0)
    if show_info >= 2:
        print(f'Searching within all files of type {file_type}')
        print(files)
    
    # search file in all files
    count_match = 0; file_names = [] # initialize counter and list for number of matches
    for file in files:
        if search_str == file[0:len(search_str)]:
            if show_info >= 0.5: print(f'Found file: {file}')
            file_names.append(file)
            count_match += 1
    
    # create file_path output or return empty string and warning/error message
    if count_match == 0: 
        print(f'ERROR. File starting with {search_str} and ending in .{file_type} not found!')
    else: 
        print(f'Found {count_match} files starting with {search_str} and ending in .{file_type}')
    #
    return file_names

#
#
#