# Copyright (c) 2023 imSauravB
#
# -*- coding:utf-8 -*-
# @Script: imageSwapAndEnhance.py
# @Author: imSauravB
# @Email: sauravkumarbehera@gmail.com
# @Create At: 2023-06-25 10:58:29
# @Last Modified By: imSauravB
# @Last Modified At: 2023-06-25 14:24:58
# @Description: This is description.

#!/usr/bin/env python3


import os

import cv2
import insightface
from os import listdir
from os.path import isfile, join
import onnxruntime
import gfpgan

providers = onnxruntime.get_available_providers()

if 'TensorrtExecutionProvider' in providers:
    providers.remove('TensorrtExecutionProvider')

FACE_ANALYSER = None
FACE_SWAPPER = None
FACE_ENHANCER = None

targetDirPath = "C:/Users/saurav/Desktop/targetDir/"
sourceImage = "C:/Users/saurav/Desktop/source.jpg"

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

def getInsightFaceAnalyser():
    global FACE_ANALYSER
    if FACE_ANALYSER is None:
        FACE_ANALYSER = insightface.app.FaceAnalysis(name='buffalo_l', providers=providers)
        FACE_ANALYSER.prepare(ctx_id=0, det_size=(640, 640))
    return FACE_ANALYSER


def getSourceFace(image):
    face = getInsightFaceAnalyser().get(image)
    try:
        return sorted(face, key=lambda x: x.bbox[0])[0]
    except IndexError:
        return None

def getAllFaces(image):
    try:
        return getInsightFaceAnalyser().get(image)
    except IndexError:
        return None

def getInsightFaceSwapper():
    global FACE_SWAPPER
    if FACE_SWAPPER is None:
        model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../faceSwapModel.onnx')
        FACE_SWAPPER = insightface.model_zoo.get_model(model_path, providers=providers)
    return FACE_SWAPPER


def swapFace(source_face, target_face, frame):
    if target_face:
        return getInsightFaceSwapper().get(frame, target_face, source_face, paste_back=True)
    return frame


def processFaces(sourceFace, targetImage):
    allFaces = getAllFaces(targetImage)
    if allFaces:
        for face in allFaces:
            targetImage = swapFace(sourceFace, face, targetImage)
        print('.', end='', flush=True)
    else:
        print('S', end='', flush=True)

    return targetImage

def processImgDir(sourceFaceImagePath, allImagesPath):
    sourceFace = getSourceFace(cv2.imread(sourceFaceImagePath))
    print("Source Face Read Done")
    for imagePath in allImagesPath:
        targetImgPath = os.path.join( targetDirPath, imagePath)
        print("Image Path: " + str(targetImgPath) )
        image = cv2.imread(targetImgPath)
        try:
            resultFirst = processFaces(sourceFace, image)
            finalResult = enhanceFace(resultFirst)
            cv2.imwrite(targetImgPath, finalResult)
        except Exception as exception:
            print(exception)
            pass

def main():
    try:
        print("##### in main #####")
        imageFiles = [f for f in listdir(targetDirPath) if isfile(join(targetDirPath, f))]
        processImgDir(sourceImage, imageFiles)
    except:
        raise

main()
