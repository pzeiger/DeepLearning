import numpy as np
from PIL import Image
import glob

def load_warwick(root_dir):
    # Loads the WARWICK dataset from png images
     
    # create list of image objects
    images = []
    labelmasks = []    
    
    for ipath, lpath in zip(glob.glob(root_dir + "/image*.png"),\
                            glob.glob(root_dir + "/label*.png")):
        
        image = Image.open(ipath)
        images.append(image)
        
        labelmask = Image.open(lpath)
        labelmasks.append(labelmask)
            
#    images = np.array(images).astype(np.double)/255.
#    labelmasks = np.array(labelmasks).astype(np.double)/255.
#    labelmasks = labelmasks.reshape(labelmasks.shape[0], labelmasks.shape[1], labelmasks.shape[2], 1)
    return images, labelmasks
