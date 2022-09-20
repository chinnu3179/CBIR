from gabor import GaborDescriptor
import argparse
import glob
import cv2

ap = argparse.ArgumentParser()

ap.add_argument("-d","--dataset", required = True , help = "Path to directory that contains images")
ap.add_argument("-i","--index_gabor", required = True , help = "Path to where the index will be stored")
args = vars(ap.parse_args())


#Initializing our color descriptor
params = {"theta":4, "frequency":(0,1,0.5,0.8), "sigma":(1,3),"n_slice":2}

gd = GaborDescriptor(params)

gaborKernels = gd.kernels()

#open the output index file for writing
output = open(args["index_gabor"],"w")

#Using global to get path of images and go through all of them
for imagePath in glob.glob(args["dataset"]+"/*.jpg"):
    #Get the UID of the image path and load the image
    imageUID =  imagePath[imagePath.rfind("/")+1:]
    image = cv2.imread(imagePath)
    #Using the describe function 
    features = gd.gaborHistogram(image,gaborKernels)
    features = [str(f) for f in features]
    print(imageUID)
    output.write("%s,%s\n" % (imageUID,",".join(features)))
    
# closing the index file
output.close()
