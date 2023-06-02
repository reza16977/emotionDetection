from deep_emotion import Deep_Emotion

from __future__ import print_function
import argparse
import numpy  as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net=Deep_Emotion()
net.load_state_dict(torch.load('DeepEmotion_trainded1.pt'))
net.to(device)


import cv2

path='haarcascade_frontalface_default.xml'
font_scale=1.5
font=cv2.FONT_HERSHEY_PLAIN

rectangle_bgr=(255,255,255)

img=np.zeros((500,500))
text='Some text '
(text_width,text_height)=cv2.getTextSize(text,font,fontScale=font_scale,thickness=1)[0]

text_offset_x=10
text_offset_y=img.shape[0]-25

box_coords=((text_offset_x,text_offset_y),(text_offset_x+text_width+2,text_offset_y-text_height-2))
cv2.rectangle(img,box_coords[0],box_coords[1],rectangle_bgr,cv2.FILLED)
cv2.putText(img,text,(text_offset_x,text_offset_y),font,fontScale=font_scale,color=(0,0,0),thickness=1)

cap=cv2.VideoCapture(1)
if not cap.isOpened():
    cap=cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError('Cannot open webcam')
    
while True:
    ret,frame=cap.read()
    faceCascade = cv2.CascadeClassifier(path)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.1, 4)
    for x, y, w, h in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            facess = faceCascade.detectMultiScale(roi_gray)

            if len(facess)==0:
                print("face not detected")
            else:
                for (ex,ey,ew,eh) in facess:
                    
                     face_roi=roi_color[ey:ey+eh,ex:ex+ew]
                    
    graytemp=cv2.cvtColor(face_roi,cv2.COLOR_BGR2GRAY)
    final_image=cv2.resize(graytemp,(48,48))
    final_image=np.expand_dims(final_image,axis=0)
    final_image=np.expand_dims(final_image,axis=0)
    final_image=final_image/255.0
    dataa=torch.from_numpy(final_image)
    dataa=dataa.type(torch.FloatTensor)
    dataa=dataa.to(device)
    outputs=net(dataa)
    Pred=F.softmax(outputs,dim=1)
    Predictions=torch.argmax(Pred)
    print(cv2.FONT_HERSHEY_SIMPLEXIM)
    
    font_scale=1.5
    font=cv2.FONT_HERSHEY_PLAIN
    
    if((Predictions==0)):
        status='Angry'
        
        x1,y1,w1,h1=0,0,175,75
        
        cv2.rectangle(frame,(x1,x1),(x1+w1,y1+h1),(0,0,0),-1)
        cv2.putText(frame,status,(x1+int(w1/10),y1+int(h1/2)),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
        cv2.putText(frame,status,(100,150),font,3,(0,0,255),2,cv2.LINE_4)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255))
        
        
    elif((Predictions==1)):
        status='Disgust'
        
        x1,y1,w1,h1=0,0,175,75
        
        cv2.rectangle(frame,(x1,x1),(x1+w1,y1+h1),(0,0,0),-1)
        cv2.putText(frame,status,(x1+int(w1/10),y1+int(h1/2)),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
        cv2.putText(frame,status,(100,150),font,3,(0,0,255),2,cv2.LINE_4)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255))
        
    elif((Predictions==2)):
        status='Fear'
        
        x1,y1,w1,h1=0,0,175,75
        
        cv2.rectangle(frame,(x1,x1),(x1+w1,y1+h1),(0,0,0),-1)
        cv2.putText(frame,status,(x1+int(w1/10),y1+int(h1/2)),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
        cv2.putText(frame,status,(100,150),font,3,(0,0,255),2,cv2.LINE_4)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255))
        
    elif ((Predictions==3)):
        status='Happy'
        
        x1,y1,w1,h1=0,0,175,75
        
        cv2.rectangle(frame,(x1,x1),(x1+w1,y1+h1),(0,0,0),-1)
        cv2.putText(frame,status,(x1+int(w1/10),y1+int(h1/2)),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
        cv2.putText(frame,status,(100,150),font,3,(0,0,255),2,cv2.LINE_4)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255))
        
    elif((Predictions==4)):
        status='Sad'
        
        x1,y1,w1,h1=0,0,175,75
        
        cv2.rectangle(frame,(x1,x1),(x1+w1,y1+h1),(0,0,0),-1)
        cv2.putText(frame,status,(x1+int(w1/10),y1+int(h1/2)),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
        cv2.putText(frame,status,(100,150),font,3,(0,0,255),2,cv2.LINE_4)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255))
        
    elif((Predictions==5)):
        status='Suprise'
        
        x1,y1,w1,h1=0,0,175,75
        
        cv2.rectangle(frame,(x1,x1),(x1+w1,y1+h1),(0,0,0),-1)
        cv2.putText(frame,status,(x1+int(w1/10),y1+int(h1/2)),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
        cv2.putText(frame,status,(100,150),font,3,(0,0,255),2,cv2.LINE_4)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255))
        
    elif((Predictions==6)):
        status='Natural'
        
        x1,y1,w1,h1=0,0,175,75
        
        cv2.rectangle(frame,(x1,x1),(x1+w1,y1+h1),(0,0,0),-1)
        cv2.putText(frame,status,(x1+int(w1/10),y1+int(h1/2)),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
        cv2.putText(frame,status,(100,150),font,3,(0,0,255),2,cv2.LINE_4)
        
        cv2.imsho('face emotion recognization',frame)
        if cv2.waitKey(2)&0xFF==ord('q'):
            break
            
cap.release()
cv2.destroyAllWindows()
