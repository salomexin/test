import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('fundus1.jpg')
sp = img.shape
img = cv2.resize(img,(sp[1]/4, sp[0]/4))

img1 = np.float32(img)
kernel1 = np.ones((25,25),np.float32)/625

kernel2 = np.ones((5,5),np.float32)/9

dst1 = cv2.filter2D(img1,-1,kernel1)
#dst2 = cv2.filter2D(img1,-1,kernel2)

I1 = img1-dst1

I2 = cv2.filter2D(I1,-1,kernel2)
I2 = cv2.GaussianBlur(I2,(5,5),0)
I1 =  np.uint8((I1-np.min(I1))/(np.max(I1)-np.min(I1))*256)

#I2 = cv2.filter2D(I1,-1,kernel2)

I2 =  np.uint8((I1-np.min(I1))/(np.max(I2)-np.min(I2))*256)

clahe=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
I10 = clahe.apply(I2[:,:,0])
I11 = clahe.apply(I2[:,:,1])
I12 = clahe.apply(I2[:,:,2])


II = np.zeros([sp[0]/4, sp[1]/4,3])
II[:,:,0] = I10
II[:,:,1] = I11
II[:,:,2] = I12

retval, I21 = cv2.threshold(255-I11, 0, 255, cv2.THRESH_OTSU) 
retval, I22 = cv2.threshold(I12, 0, 255, cv2.THRESH_OTSU)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  

T1=cv2.morphologyEx(I21, cv2.MORPH_OPEN, kernel)
T2=cv2.morphologyEx(I22, cv2.MORPH_CLOSE, kernel)

cv2.imshow("1",img)
cv2.imshow("2",I1)
cv2.imshow("3",I2)
cv2.imshow("4",I10)
cv2.imshow("5",I11)
cv2.imshow("8",I21)
cv2.imshow("6",I12)
cv2.imshow("9",I22)
cv2.imshow("7",np.uint8(II))
cv2.imshow("10",T1)
cv2.imshow("11",T2)

cv2.waitKey(0)
#plt.subplot(1,2,1),plt.imshow(img1,'gray')
#plt.subplot(1,2,2),plt.imshow(dst,'gray')
