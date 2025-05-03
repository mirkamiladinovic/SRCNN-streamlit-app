import cv2
import numpy as np
from .metrics import compare_images

def degrade_image(img, factor=2):
    h, w = img.shape[:2]
    new_h, new_w = int(h / factor), int(w / factor)

    # Downscale and upscale
    img_down = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    img_up = cv2.resize(img_down, (w, h), interpolation=cv2.INTER_LINEAR)

    return img_up


def modcrop(img, scale):
  tmpsz = img.shape
  sz = tmpsz[0:2]
  sz = sz - np.mod(sz, scale)
  img = img[0:sz[0], 1:sz[1]]
  return img

def shave(image, border):
  img = image[border: -border, border: -border]
  return img

def predict(image_path, model):
    ref = cv2.imread(image_path)
    degraded = degrade_image(ref, factor=2)

    degraded = modcrop(degraded, 3)
    ref = modcrop(ref, 3)

    ycrcb = cv2.cvtColor(degraded, cv2.COLOR_BGR2YCrCb)
    y = ycrcb[:, :, 0].astype(np.float32) / 255.0
    input_data = np.expand_dims(np.expand_dims(y, axis=0), axis=-1)

    output_data = model.predict(input_data)[0, :, :, 0] * 255
    output_data = np.clip(output_data, 0, 255).astype(np.uint8)

    ycrcb = shave(ycrcb, 6)
    ycrcb[:, :, 0] = output_data
    result = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    ref = shave(ref, 6)
    degraded = shave(degraded, 6)

    scores_degraded = compare_images(degraded, ref)
    scores_output = compare_images(result, ref)

    return ref, degraded, result, scores_degraded, scores_output
