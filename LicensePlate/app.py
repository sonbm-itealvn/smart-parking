import cv2
from detection import detection
def app(source):
    count = 0
    track_id = 0
    tracking_dict = {}
    cap = cv2.VideoCapture(source)
    while cap.isOpened(): 
        count += 1
        frameName = r"tests\frame.jpg"
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(frameName,frame)
            tracking_dict, count, track_id = detection("frame.jpg",
                                                    tracking_dict,
                                                    count,
                                                    track_id)
            if 0xFF == ord('q'):
                break
        else:
            print(tracking_dict)
            return 0
    
    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    app(r"test_footage\parking_test6.mp4")