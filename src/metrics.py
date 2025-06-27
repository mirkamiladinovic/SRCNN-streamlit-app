import numpy as np
import math
from skimage.metrics import structural_similarity as ssim

def psnr(target, ref):
  target_data = target.astype(float)
  ref_data = ref.astype(float)

  diff = ref_data - target_data
  diff = diff.flatten('C')

  rmse = math.sqrt(np.mean(diff ** 2.))

  return 20 * math.log10(255. / rmse) if rmse != 0 else float('inf')

def mse(target, ref):
  err = np.sum((target.astype(float) - ref.astype(float)) ** 2)
  err /= float(target.shape[0] * target.shape[1])

  return err

def compare_images(target, ref):
  scores = []
  scores.append(psnr(target, ref))
  scores.append(mse(target, ref))
  scores.append(ssim(target, ref, channel_axis=-1))

  return scores
