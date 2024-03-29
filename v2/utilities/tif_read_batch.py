# Batch conversion of .tif images generated on Google Earth to .npy arrays

import os
from osgeo import gdal
import numpy as np
import sys, getopt


os.chdir("F:/gis/github/v2/image_processing")
base_path = os.getcwd()
in_path = os.path.join(base_path, "raw_tifs/")
city = "Zaragoza" # change this to different city names
multi_tiled = False # supports large multi-tiled geotiff
gcs = False

out_path = os.path.join(base_path, "output/")

tile_size_x = 1024
tile_size_y = 1024

print(f"Read in images from {out_path} for {city}")

def convert(input,fname,outpath):
    input = gdal.Open(input)
    input = input.ReadAsArray()

    n_c = input.shape[0]
    end_x = input.shape[1]
    end_y = input.shape[2]
    nearest_x = int(np.floor(end_x/tile_size_x)) #nearest complete tiles at x axis
    nearest_y =int(np.floor(end_y/tile_size_y)) #nearest complete tiles at y axis
    #manual padding
    newimg = np.zeros((n_c,(nearest_x+1)*tile_size_x, (nearest_y+1)*tile_size_y))
    newimg[:,:end_x, :end_y] = input[:,:end_x, :end_y]
    # newimg = np.reshape(newimg,(-1,n_c,tile_size_x, tile_size_y)) -  contiguous array reshaping, wrong
    new_end_x = newimg.shape[1]
    new_end_y = newimg.shape[2]
    tiler = 0
    for x in range(0, new_end_x//tile_size_x):
        for y in range(0, new_end_y//tile_size_y):
            arr = newimg[:, (x*tile_size_x): ((x+1)*tile_size_x), (y*tile_size_y): ((y+1)*tile_size_y)]
            print(f"Cropping: {(x*tile_size_x)}:{((x+1)*tile_size_x)}, {(y*tile_size_y)}:{((y+1)*tile_size_y)}")
            np.save(os.path.join(out_path ,  f"{fname}_{str(tiler)}.npy"),arr)
            tiler += 1


for root, directories, files in os.walk(in_path):
    for file in files:
        if file.lower().endswith(".tif"):
            input = root+file
            print(f"found {input}")
            if gcs:
                if multi_tiled:
                    names = file.split("_")
                    fname = f'{names[0]}_{names[1]}_{names[2].split(".")[0]}'
                else:
                    fname = file.split(".")[0] #if saving to gcs, file names have to be manually specific so no reformatting needed
            else:
                fname = city + "_" + file.split("_")[2][0:4]
                if multi_tiled:
                    fname = f'{city}_{file.split("_")[2][0:4]}_{file.split("-")[1]}-{file.split("-")[2].split(".")[0]}'
            print(f"writing {fname}")
            convert(input, fname, out_path)

print("Done")
