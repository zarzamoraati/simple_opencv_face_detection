import pickle
import os
import cv2
from train_model import generate_embeddings,training_model
from keras.api.applications.vgg16 import VGG16



model=VGG16(include_top=False,weights="imagenet",input_shape=(160,160,3))
path=os.getenv("FACES")

def main(path:str,model_name:str):
    
    cap=cv2.VideoCapture(os.getenv("STREAM_IP"))
    face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    if not cap.isOpened():
        print('Stream is not open')
        exit()
        
    classifier=None

    try:
        with open(model_name,'rb') as f:

            classifier=pickle.load(f)
    except:
        training_model(path,model_name=model_name)
        with open(model_name,'rb') as f:
        
            classifier = pickle.load(f)
    
    while True:
        status,frame=cap.read()
        if not status:
            break
        
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces=face_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=4)
        
        for (x,y,w,z) in faces:
            face=frame[y:y+z,x:x+w]
            face=cv2.resize(face,(160,160))
            embedding=generate_embeddings(model=model,img_frame=face)
            label_pred=classifier.predict([embedding])
            
            cv2.putText(frame,label_pred[0],(x,y-10),cv2.FONT_HERSHEY_COMPLEX,0.9,(0,255,0))
            cv2.rectangle(frame,(x,y),(x+w,y+z),(0,255,0),2)

            cv2.imshow('stream',frame)
            if cv2.waitKey(1) == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()

        
    
    
main(path=path,model_name='classifier_model.pkl')
    


