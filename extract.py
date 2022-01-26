import zipfile, os

working_directory = '/home/DIINF/labello/U-net-original/experimentos/imagenes/'
os.chdir(working_directory)

for file in os.listdir(working_directory):   # get the list of files
    print(file)
    if zipfile.is_zipfile(file): # if it is a zipfile, extract it
        with zipfile.ZipFile(file) as item: # treat the file as a zip
           item.extractall()  # extract it in the working directory