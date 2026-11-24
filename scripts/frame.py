# The picture as the camera gives it to the network: stretched to W x H. Used
# for calibration, the gate and the operating points alike.
import cv2


def rgb(path, w, h):
    im = cv2.imread(path)
    if im is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR),
                        cv2.COLOR_BGR2RGB)
