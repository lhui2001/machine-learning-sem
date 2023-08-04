# add and reorganize columns in txt dataframe
import pandas as pd
import os
import numpy as np

def main():
    
    path = "C:/.../AttU_Net_box/"
            
    # iterate through txt files of the folder
    for file in os.listdir(path):

        # create the full input path and read the image
        input_path = os.path.join(path, file)
        df = pd.read_csv(input_path, sep=" ", header=None)
        
        # add column
        df[len(df.columns)] = ["object"] * len(df.index)
        df[len(df.columns)] = [0.80] * len(df.index)
        
        # reorganize columns
        cols = df.columns.tolist()    
        cols = cols[-1:] + cols[:-1]
        df = df[cols] 
        
        # folder names
        name = os.path.splitext(os.path.basename(file))[0]

        # export values
        np.savetxt(f"C:/.../AttU_Net_box/{name}.txt", df.values, fmt='%s')
                    
if __name__ == '__main__':
    main



# delete and reorganize columns in txt dataframe
import pandas as pd
import os
import numpy as np

def main():
    
    path = "C:/.../AttU_Net_box/"
            
    # iterate through excel files of the folder
    for file in os.listdir(path):

        # create the full input path and read the image
        input_path = os.path.join(path, file)
        df = pd.read_csv(input_path, sep=" ", header=None)
        
        # delete columns
        df = df.drop(df.columns[[0, 7]], axis=1)
        
        # folder names
        name = os.path.splitext(os.path.basename(file))[0]

        # export values
        np.savetxt(f"C:/.../AttU_Net_box/{name}.txt", df.values, fmt='%s')
                    
if __name__ == '__main__':
    main()
