import cv2
import os
from dotenv import load_dotenv

load_dotenv()

def extract_captures(output_dir:str):
    cap = cv2.VideoCapture(os.getenv("STREAM_IP"))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    if not cap.isOpened():
        print("Capture is not opened")
        exit()
    count_frame=0
    while True:
        state,frame=cap.read()
        if not state:
            break
        #resize_img=cv2.resize(frame,(400,400))
        cv2.imwrite(f'{output_dir}/capture_{count_frame}.jpg',frame)
        if cv2.waitKey(1)==ord('q'):
            break

        count_frame+=1
    cap.release()
    cv2.destroyAllWindows()

    return output_dir


    
        

output_dir=os.getenv("CAPTURES")

extract_captures(output_dir=output_dir)