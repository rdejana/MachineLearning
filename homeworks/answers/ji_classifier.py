#!/usr/bin/python
#
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#

import jetson.inference
import jetson.utils

import argparse
import sys
import time

# parse the command line
parser = argparse.ArgumentParser(description="Classify an image using an image recognition DNN.", 
						   formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.imageNet.Usage())

parser.add_argument("file_in", type=str, help="filename of the input image to process")
parser.add_argument("file_out", type=str, default=None, nargs='?', help="filename of the output image to save")
parser.add_argument("--network", type=str, default="googlenet", help="pre-trained model to load (see below for options)")

try:
	opt = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)

# load an image (into shared CPU/GPU memory)
img, width, height = jetson.utils.loadImageRGBA(opt.file_in)

# load the recognition network
net = jetson.inference.imageNet(opt.network, sys.argv)

# classify the image
for _ in range(5):
	start = time.perf_counter()
	class_idx, confidence = net.Classify(img, width, height)
	inference_time = time.perf_counter() - start
	print('%.1fms' % (inference_time * 1000))
# find the object description
class_desc = net.GetClassDesc(class_idx)

# print out the result
print("image is recognized as '{:s}' (class #{:d}) with {:f}% confidence\n".format(class_desc, class_idx, confidence * 100))


