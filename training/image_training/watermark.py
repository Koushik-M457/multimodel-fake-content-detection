import cv2
import numpy as np

def watermark_score(image_path):
    """
    Returns a score between 0 and 1
    Higher = more likely AI-generated
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Edge detection
    edges = cv2.Canny(img, 100, 200)
    edge_density = np.mean(edges) / 255.0

    # Frequency domain (FFT)
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude = np.log(np.abs(fshift) + 1)
    freq_score = np.mean(magnitude) / 10.0

    # Normalize
    wm_score = min(1.0, (0.6 * edge_density + 0.4 * freq_score))
    return wm_score
