import os
def move1lvup(drt:str):
    '''
    Description:
    Move files one level up, retain subfolder names in file names
    
    Input
    drt (str)
        -Target directory where there are subfolders nested within with images

    Output
    None
    '''
    path = os.walk(drt)
    for root, directories, files in path:
        for directory in directories:
            #print(directory) # current dir
            target = drt+"/"+str(directory)
            
            for r,d, files in os.walk(target):
                for f in files:
                    os.rename(target+"/"+f, drt+"/"+str(directory)+"-"+f)
            os.rmdir(target)

        