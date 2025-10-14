from ocr import recogFunc
from ultralytics import YOLO
from crop import cropFunc2, drawFrame
from licenseTracking import trackingFunc
import numpy as np

model = YOLO(r"models\best.pt")

def detection(img,tracking_dict,count,track_id):
    res = model("tests/" + img, conf=0.6)
    crops = res[0].boxes.numpy()
    
    for j in range(len(crops)):
        l = list(crops.data[j])
        drawFrame(int(l[0]), int(l[1]), int(l[2]), int(l[3]),img)
        check, tracking_dict, count, track_id = trackingFunc(tracking_dict,
                                                            l,
                                                            count,
                                                            track_id)
        
        if check==False:
            frame = cropFunc2("tests/" + img, l)
            if frame!=[]:
                img2 = r"tests\scaledLP.jpg"
                frame.save(img2)
                recogFunc(np.array(frame))
    return tracking_dict, count, track_id