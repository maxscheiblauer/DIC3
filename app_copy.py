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
from PIL import ImageFont, ImageOps, ImageColor
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
from flask import Flask, make_response, request, Response, jsonify
import random
import re
import base64



app = Flask(__name__)

#Load detector model

model_path = r"https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1"
detector = hub.load(model_path).signatures['default']

#Functions for Box-Drawing

def draw_bounding_box_on_image(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               color,
                               font,
                               thickness=4,
                               display_str_list=()):
    """Adds a bounding box to an image."""
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                ymin * im_height, ymax * im_height)
    draw.line([(left, top), (left, bottom), (right, bottom), (right, top),
                (left, top)],
                width=thickness,
                fill=color)

    # If the total height of the display strings added to the top of the bounding
    # box exceeds the top of the image, stack the strings below the bounding box
    # instead of above.
    display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
    # Each display_str has a top and bottom margin of 0.05x.
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = top + total_display_str_height
    # Reverse list and print from bottom to top.
    for display_str in display_str_list[::-1]:
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)
        draw.rectangle([(left, text_bottom - text_height - 2 * margin),
                        (left + text_width, text_bottom)],
                       fill=color)
        draw.text((left + margin, text_bottom - text_height - margin),
                  display_str,
                  fill="black",
                  font=font)
        text_bottom -= text_height - 2 * margin

def draw_boxes(image, boxes, class_names, scores, max_boxes=10, min_score=0.1):
    """Overlay labeled boxes on an image with formatted scores and label names."""
    colors = list(ImageColor.colormap.values())

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Regular.ttf",
                              25)
    except IOError:
        print("Font not found, using default font.")
        font = ImageFont.load_default()

    for i in range(min(boxes.shape[0], max_boxes)):
        if scores[i] >= min_score:
            ymin, xmin, ymax, xmax = tuple(boxes[i])
            display_str = "{}: {}%".format(class_names[i].decode("ascii"),
                                         int(100 * scores[i]))
            color = colors[hash(class_names[i]) % len(colors)]
            image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
            draw_bounding_box_on_image(
              image_pil,
              ymin,
              xmin,
              ymax,
              xmax,
              color,
              font,
              display_str_list=[display_str])
            np.copyto(image, np.array(image_pil))
    return image

#to encode the resulting back to base64

def encode_image(image):
    img_byte_arr = io.BytesIO()
    Image.fromarray(image).save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    img_base64 = base64.b64encode(img_byte_arr).decode('utf-8')
    return img_base64

#only include top 10 detections
def shorten_array(arr):
    if len(arr) > 10:
        arr = arr[:10] 
    return arr


def detection_loop(images, upload_time):
   
  #initilaize result dict
  detection_results = {
    "detection_class_entities": [],
    "detection_scores": [],
    "detection_boxes": [],
    "inference_times": [],
    "image_with_boxes": []}
   

  for img in images:
       
      #preparing and normalizing image
      normalized_img = (img - img.min()) / (img.max() - img.min())
      img_tensor_1 = tf.convert_to_tensor(normalized_img, dtype=tf.float32)
      img_tensor = tf.image.convert_image_dtype(img_tensor_1, dtype=tf.float32)[tf.newaxis, ...]

      #run detector and time it
      start_inf_time = time.time()
      result = detector(img_tensor)
      end_inf_time = time.time()
      inf_time = end_inf_time - start_inf_time

      #create intermediary dictionary from result
      result = {key:value.numpy() for key,value in result.items()}

      entities = shorten_array(result["detection_class_entities"])
      scores = shorten_array(result["detection_scores"])
      boxes = shorten_array(result["detection_boxes"])

      #draw boxes on image
      image_with_boxes = draw_boxes(img, result["detection_boxes"],
                                    result["detection_class_entities"], result["detection_scores"])
      
      #filling the result dictionary
      detection_results["detection_class_entities"].append(entities)
      detection_results["detection_scores"].append(scores)
      detection_results["detection_boxes"].append(boxes)
      detection_results["inference_times"].append(inf_time)
      detection_results["image_with_boxes"].append(image_with_boxes)


  detection_results["avg_inference_time"] = np.mean(detection_results["inference_times"])
  detection_results["upload_time"] = upload_time
  return make_response(jsonify(detection_results), 200)



#initializing the flask app
app = Flask(__name__)

#routing http posts to this method
@app.route('/api/detect', methods=['POST', 'GET'])
def main():
  start_upload_time = time.time()
  data=  request.get_json(force = True)
  #get the array of images from the json body
  imgs = data['images']
  end_upload_time = time.time()
  upload_time = end_upload_time - start_upload_time
 
  #prepare images for object detection 
  images =[]
  for img in imgs:
    images.append((np.array(Image.open(io.BytesIO(base64.b64decode(img))),dtype=np.float32)))
  
  
  return detection_loop(images, upload_time)
  
# status_code = Response(status = 200)
#  return status_code
# image=cv2.imread(args.input)
# image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

if __name__ == '__main__':
    app.run(debug = True, host = '0.0.0.0')
