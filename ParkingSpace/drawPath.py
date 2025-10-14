from distances import euclid_dist, centroid_calc
import cv2
def path(img,start,spot):
    landmarks = {0:start,
                1:[100,100],
                2:[80,165],
                3:[60,300]}
    spot_check = centroid_calc(spot)[0], int(spot[1])
    best_landmark = 0
    min_dist = euclid_dist(spot_check, landmarks[best_landmark])

    for i in landmarks.keys():
        cv2.circle(img, landmarks[i], 4, (0, 255, 0), -1)
        dist = euclid_dist(spot_check, landmarks[i])
        if dist<min_dist:
            min_dist = dist
            best_landmark = i
    print(best_landmark)
    for i in landmarks.keys():
        if i==0:
            continue
        if i>best_landmark:
            break
        img = cv2.line(img, landmarks[i-1], landmarks[i], (0, 255, 0),3)
    img = cv2.line(img, landmarks[best_landmark], spot_check, (0, 255, 0),3)
    return img