import cv2
from distances import euclid_dist, centroid_calc
from drawPath import path   
start = [110,5]
spot = [377.52,168.17,395.5,210.34]

frame = cv2.imread(r'tests\drawFrame.jpg')
landmarks = [[100,100],[80,165],[60,300]]

# cv2.circle(frame, start, 4, (0, 0, 255), -1)
# for i in landmarks:
#     cv2.circle(frame, i, 4, (0, 0, 255), -1)

frame = path(frame,start,spot)
cv2.imwrite(r'tests\drawFrame2.jpg',frame)