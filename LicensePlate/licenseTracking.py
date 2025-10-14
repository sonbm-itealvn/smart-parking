import math
from ocr import recogFunc
from crop import cropFunc2
import numpy as np
def centroid_calc(x, y, w, h):
    return int((x + x + w) / 2), int((y + y + h) / 2)

def euclid_dist(pt1,pt2):
    return math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])

def trackingFunc(tracking_dict, l, count, track_id):
    check = False
    center_points_cur_frame = []
    x, y, w, h = int(l[0]), int(l[1]), int(l[2]), int(l[3])
    cx, cy = centroid_calc(x,y,w,h)
    if h>245:
        center_points_cur_frame.append((cx, cy))
        center_points_prev_frame = list(tracking_dict.values())
        if count <= 2:
            for pt in center_points_cur_frame:
                for pt2 in center_points_prev_frame:
                    distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])
                    if distance < 20:
                        tracking_dict[track_id] = pt
                        track_id += 1
        else:
            tracking_dict_copy = tracking_dict.copy()
            center_points_cur_frame_copy = center_points_cur_frame.copy()
            for object_id, pt2 in tracking_dict_copy.items():
                object_exists = False
                for pt in center_points_cur_frame_copy:
                    distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])

                    # Update IDs position
                    if distance < 20:
                        tracking_dict[object_id] = pt
                        object_exists = True
                        if pt in center_points_cur_frame:
                            center_points_cur_frame.remove(pt)
                        continue

                # Remove IDs lost
                if not object_exists:
                    check = True
                    tracking_dict.pop(object_id)

            # Add new IDs found
            for pt in center_points_cur_frame:
                # frame = cv2.imread('tests/frame.jpg')
                # frame = frame[x:x+w,y:y+h]
                lp = cropFunc2('tests/frame.jpg',(x,y,w,h))
                recogFunc(np.array(lp))
                check = True
                tracking_dict[track_id] = pt
                track_id += 1
                

        # frame = cv2.imread('tests/drawFrame.jpg')
        # for object_id, pt in tracking_dict.items():
        #     cv2.circle(frame, pt, 5, (0, 0, 255), -1)
        #     cv2.putText(frame, str(object_id), (pt[0], pt[1] - 7), 0, 1, (0, 0, 255), 2)
        # cv2.imwrite('tests/drawFrame.jpg',frame)

    return check, tracking_dict, count, track_id
