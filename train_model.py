import cv2
import os
from keras.api.applications.vgg16 import VGG16
import numpy as np
from  sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle



model=VGG16(include_top=False, weights='imagenet', input_shape=(160,160,3))

def process_faces(path:str):
    if not os.path.exists(path):
        print('Invalid Route: ',path)
        exit()
    X,y = [], []
    for label in os.listdir(path):
        for file in os.listdir(f'{path}/{label}'):
            img_path=f'{path}/{label}/{file}'
            img_frame=cv2.imread(img_path)
            img_frame=cv2.resize(img_frame,(160,160))
            embedding=generate_embeddings(model,img_frame)
            X.append(embedding)
            y.append(label)
            print(label)
    return X,y

    
    
def generate_embeddings(model,img_frame:np.ndarray):
    # Precision
    img_frame=img_frame.astype("float32")
    mean,std = img_frame.mean(),img_frame.std()
    ## Normalization
    img_frame = (img_frame - mean) / std
    ## Reshape form 
    img_frame = np.expand_dims(img_frame,0) 
    embedding=model.predict(img_frame)
    embedding=embedding.flatten()
    embedding=embedding.reshape(1,-1)
 
    return embedding[0]



def training_model(path:str,model_name:str="classifier.pkl"):
    X,y=process_faces(path=path)

    x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=40)

    classifier=SVC(kernel='linear',probability=True)
    classifier.fit(x_train,y_train)
    prediction=classifier.predict(x_test)
    print(accuracy_score(y_test,prediction))
    
    with open(model_name,"wb") as f:
        pickle.dump(classifier,f)
        
    




    