import math

def centroid_calc(x):
    return int((x[0] + x[2]) / 2), int((x[1] + x[3]) / 2)

def euclid_dist(pt1,pt2):
    return math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])