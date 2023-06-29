# Copyright (c) 2023 imSauravB
#
# -*- coding:utf-8 -*-
# @Script: imageEnhance.py
# @Author: imSauravB
# @Email: sauravdakhinray@gmail.com
# @Create At: 2023-06-29 01:36:28
# @Last Modified By: imSauravB
# @Last Modified At: 2023-06-29 11:30:42
# @Description: This is description.



import os
import cv2
from os import listdir
from os.path import isfile, join
import gfpgan
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

SCALE = 4
FACE_ENHANCER = None


targetDirPath = "C:/Users/saurav/Desktop/avatar/"

upsampler = RealESRGANer(
    scale = SCALE,
    model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../imageUpscaleModel-x4plus.pth'),
    dni_weight = None,
    #model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu'),
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4),
    tile = 0,
    tile_pad = 10,
    pre_pad = 0)

def getFaceEnhancer():
    global FACE_ENHANCER
    if FACE_ENHANCER is None:
        model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../faceEnhanceModel.pth')
        # todo: set models path https://github.com/TencentARC/GFPGAN/issues/399
        FACE_ENHANCER = gfpgan.GFPGANer(model_path=model_path, upscale=SCALE, arch='clean', channel_multiplier=2, bg_upsampler=upsampler) # type: ignore[attr-defined]
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
