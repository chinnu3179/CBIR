import argparse
import cv2
import numpy as np
import h5py

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from color import ColorDescriptor
from gabor import GaborDescriptor

import Searcher

ap = argparse.ArgumentParser()
ap.add_argument("-q", "--query", required = True,help = "Path to the query image")
ap.add_argument("-c", "--class", required = True,help = "Feature class")
args = vars(ap.parse_args())

if args["class"] == "color" or args["class"] == "gabor" or args["class"] == "hog":

	if(args["class"] == "color"):
		bins = (8,12,3)
		cd = ColorDescriptor(bins)

		query = cv2.imread(args["query"])
		features = cd.describe(query)

		searcher = Searcher.Searcher("index_color.csv")
		results = searcher.search(features)

	elif args["class"] == "gabor":
		params = {"theta":4, "frequency":(0,1,0.5,0.8), "sigma":(1,3),"n_slice":2}
		gd = GaborDescriptor(params)
		gaborKernels = gd.kernels()

		query = cv2.imread(args["query"])
		features = gd.gaborHistogram(query,gaborKernels)

		searcher = Searcher.Searcher("index_gabor.csv")
		results = searcher._gsearch(features)


	myWindow = cv2.resize(query,(960,960))
	cv2.imwrite('result/{}.jpg'.format('0_query'), myWindow)
	cv2.waitKey(0)
	for (score,resultId) in results:
		print(score)
		result = cv2.imread(resultId)
		print(resultId)
		myWindow = cv2.resize(result,(480,480))
		cv2.imwrite('result/{}.jpg'.format(str(score)), myWindow)
		ch = cv2.waitKey(0)
		if ch == ord('q'):
			pass
	

