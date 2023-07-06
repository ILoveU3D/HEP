import numpy as np
import cv2

def process(image):
    image[np.where(image<0)] = 0
    return image

def histEq(images):
    results = []
    hist = np.zeros(256)
    for image in images:
        hist += np.histogram(image.flatten(), bins=256, range=[0, 256])[0]
    cdf = hist.cumsum()
    cdf = cdf / cdf.max()
    for image in images:
        max_val = np.max(image.flatten())
        min_val = np.min(image.flatten())
        image = np.floor((image - min_val) / (max_val - min_val) * 255)
        image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.LUT(image, cdf * 255).astype(np.uint8)
        image = np.asarray(image).astype("float32")
        image = image/255*(max_val-min_val)+min_val
        results.append(image)
    return results