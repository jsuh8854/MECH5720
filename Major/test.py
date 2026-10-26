import cv2
print("OpenCV version:", cv2.__version__)
print("Has optflow (contrib):", hasattr(cv2, "optflow"))
if hasattr(cv2, "optflow"):
    print([n for n in dir(cv2.optflow) if "TVL1" in n])
print([n for n in dir(cv2) if "TVL1" in n])