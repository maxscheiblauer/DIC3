# Lint as: python3
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from PIL import Image
from PIL import ImageDraw
import os
import detect
import tensorflow as tf
import tensorflow_hub as hub
import platform
import datetime
import cv2
import time
import numpy as np
import io
from io import BytesIO
from flask import Flask, request, Response, jsonify
import random
import re
import base64



app = Flask(__name__)

#Load detector model

model_path = r"https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1"
detector = hub.load(model_path).signatures['default']

def detection_loop(images):
   
    #initialize global detections
    det = {"status" : 200, "inf_time" : [], "avg_inf_time" : 0,"uploaod_time" : [], "avg_upload_time" : 0, "entities" : [], "scores" : [], "bounding_boxes" : []}

    #maximum detection per picture
    max_length = 10

    #times
    inf_times = []
    upload_times = []

    for i in images:
        #convert image to tensor

        #converted_img  = tf.image.convert_image_dtype(i, tf.float32)[tf.newaxis, ...]    WORKS TOO
        converted_image = tf.convert_to_tensor(i,tf.float32)[tf.newaxis, ...]

        #run detector and time it

        start_time = time.time()
        result = detector(converted_image)
        end_time = time.time()
        inf_time = end_time - start_time
        inf_times.append(inf_time)

        #build intermediate result dictionary NOT FINAL
        result = {key:value.numpy() for key,value in result.items()}

        #byte to string because byte does not work with json
        aux_ent = [byte.decode() for byte in result["detection_class_entities"][:max_length].tolist()]

        #add results to global detections
        det["entities"].extend(aux_ent)
        det["scores"].extend(result["detection_scores"][:max_length].tolist())
        det["bounding_boxes"].extend(result["detection_boxes"][:max_length].tolist())
    
    det["inf_time"].extend(inf_times)
    det["avg_inf_time"] = np.mean(inf_times)
    det["upload_time"].extend(upload_times)
    det["avg_upload_time"] = np.mean(upload_times)

    return make_response(jsonify(det),200)


    """
    data = {
        "status": 200,
        "bounding_boxes": bounding_boxes,
        "inf_time": inf_times,
        "avg_inf_time": str(avg_inf_time),
        "upload_time": upload_times,
        "avg_upload_time": str(avg_upload_time),
        
    }
    return make_response(jsonify(data), 200)
    """

#initializing the flask app
app = Flask(__name__)

#routing http posts to this method
@app.route('/api/detect', methods=['POST', 'GET'])
def main():
  data=  request.get_json(force = True)
  #get the array of images from the json body
  imgs = data['images']
 
  #TODO prepare images for object detection 
  #below is an example
  images =[]
  for img in imgs:
    images.append((np.array(Image.open(io.BytesIO(base64.b64decode(img))),dtype=np.float32)))
  
  
  return detection_loop(images)
  
# status_code = Response(status = 200)
#  return status_code
# image=cv2.imread(args.input)
# image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

if __name__ == '__main__':
    app.run(debug = True, host = '0.0.0.0')
