import cv2 
import os
import numpy as np

def extract_faces(directory:str,output_dir):
    if not os.path.exists(directory):
        print("Given directory not exist")
        exit(1)

    for root, _,files in os.walk(directory):
        
        label = os.path.basename(root)
        print(label)
        frame_count=0
        for file_name in files:
            recognize_faces(root,file_name,output_dir=output_dir,label=label,frame_count=frame_count)
            frame_count+=1
            
def recognize_faces(root,file_name,output_dir:str,label:str,frame_count:int):
    
    path_frame=os.path.join(root,file_name)
    face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
    frame=cv2.imread(path_frame)
    faces=face_cascade.detectMultiScale(frame,scaleFactor=1.2,minNeighbors=4)
    
    if not os.path.exists(f'{output_dir}/{label}'):
        os.makedirs(f'{output_dir}/{label}')

    
    for (x,y,w,z) in faces:
        frame=frame[y:y+z,x:x+w]
        if isinstance(frame,np.ndarray) and len(frame)>0:
            print(frame)
            cv2.imwrite(f'{output_dir}/{label}/{label}_{frame_count}.jpg',frame)


path=os.getenv("CAPTURES")
output_dir=os.getenv("FACES")


extract_faces(directory=path,output_dir=output_dir)