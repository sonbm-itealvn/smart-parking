from ultralytics import YOLO
from drawPath import path
from PIL import Image
import sys
from distances import euclid_dist, centroid_calc
import cv2
model = YOLO(r"models\best.pt")

def detection(img):
    #assuming this point to be the entrance to the parking lot
    start = (110,5)
    res = model("tests/" + img, conf=0.6)
    crops = res[0].boxes.numpy()
    for r in res:
        im_array = r.plot(conf=False,labels=False) 
        im = Image.fromarray(im_array[..., ::-1])
        im.save('tests/drawFrame.jpg')
    best_space = []
    min_dist = sys.maxsize
    for i in crops.data:
        #chk if space is empty
        if i[5]==0:
            cx, cy = centroid_calc(i[:4])
            dist = euclid_dist(start, (cx,cy))
            if dist<min_dist:
                min_dist = dist
                best_space = i[:4]
    if len(best_space)==0:
        print("All parking spaces are full")
    else:
        frame = cv2.imread(r'tests\drawFrame.jpg')
        frame = path(frame,start,best_space)
        cv2.imwrite(r'tests\drawFrame2.jpg',frame)


