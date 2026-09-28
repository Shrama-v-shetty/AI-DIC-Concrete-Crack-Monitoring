import cv2
import numpy as np

def zncc(template, window):

    template = template.astype(np.float32)
    window = window.astype(np.float32)

    template_mean = np.mean(template)
    window_mean = np.mean(window)

    numerator = np.sum(
        (template - template_mean) *
        (window - window_mean)
    )

    denominator = np.sqrt(
        np.sum((template-template_mean)**2) *
        np.sum((window-window_mean)**2)
    )

    if denominator == 0:
        return -1

    return numerator / denominator