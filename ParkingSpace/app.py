import cv2
from detection import detection

def app(source):
    cap = cv2.VideoCapture(source)
    while cap.isOpened(): 
        frameName = r"tests\frame.jpg"
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(frameName,frame)
            detection("frame.jpg")

            if 0xFF == ord('q'):
                break
        else:
            return 0
    
    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    app(r"test_footage\parking_test2.mp4")