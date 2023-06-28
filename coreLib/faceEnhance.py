# Copyright (c) 2023 imSauravB
#
# -*- coding:utf-8 -*-
# @Script: faceEnhance.py
# @Author: imSauravB
# @Email: sauravkumarbehera@gmail.com
# @Create At: 2023-06-28 23:20:56
# @Last Modified By: imSauravB
# @Last Modified At: 2023-06-28 23:33:24
# @Description: This is description.


import os
import cv2
from os import listdir
from os.path import isfile, join
import gfpgan

FACE_ENHANCER = None


targetDirPath = "C:/Users/saurav/Desktop/avatar/"

def getFaceEnhancer():
    global FACE_ENHANCER
    if FACE_ENHANCER is None:
        model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../faceEnhanceModel.pth')
        # todo: set models path https://github.com/TencentARC/GFPGAN/issues/399
        FACE_ENHANCER = gfpgan.GFPGANer(model_path=model_path, upscale=1) # type: ignore[attr-defined]
    return FACE_ENHANCER

def enhanceFace(tempFrame):
    _, _, tempFrame = getFaceEnhancer().enhance(
        tempFrame,
        paste_back=True
    )
    return tempFrame

def processImgDir(allImagesPath):
    print("Source Face Read Done")
    for imagePath in allImagesPath:
        targetImgPath = os.path.join( targetDirPath, imagePath)
        print("Image Path: " + str(targetImgPath) )
        image = cv2.imread(targetImgPath)
        try:
            finalResult = enhanceFace(image)
            enhancedTargetImgPath = targetImgPath.replace(".", "-enhanced.")
            cv2.imwrite(enhancedTargetImgPath, finalResult)
        except Exception as exception:
            print(exception)
            pass

def main():
    try:
        print("##### in main #####")
        imageFiles = [f for f in listdir(targetDirPath) if isfile(join(targetDirPath, f))]
        processImgDir(imageFiles)
    except:
        raise

main()
