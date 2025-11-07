import cv2 as cv

img1 = cv.imread('imgs/1.jpg')
img2 = cv.imread('imgs/2.jpg')
img3 = cv.imread('imgs/3.jpg')
img4 = cv.imread('imgs/4.jpg')

#img1
cv.imshow('img1',img1)
cv.waitKey(0)

__, mask1 = cv.threshold(img1, 30, 255, cv.THRESH_BINARY)
cv.imshow('mask1',mask1)
cv.waitKey(0)

mask1gray = cv.cvtColor(mask1,cv.COLOR_BGR2GRAY)
cv.imshow('mask1gray',mask1gray)
cv.waitKey(0)

__, mask1_th = cv.threshold(mask1gray, 240, 255, cv.THRESH_BINARY)
cv.imshow('mask1_th',mask1_th)
cv.waitKey(0)

mask1_inv = cv.bitwise_not(mask1_th)
cv.imshow('mask1_inv',mask1_inv)
cv.waitKey(0)

#img2
cv.imshow('img2',img2)
cv.waitKey(0)

__, mask2 = cv.threshold(img2, 30, 255, cv.THRESH_BINARY)
cv.imshow('mask2',mask2)
cv.waitKey(0)

mask2gray = cv.cvtColor(mask2,cv.COLOR_BGR2GRAY)
cv.imshow('mask2gray',mask2gray)
cv.waitKey(0)

__, mask2_th = cv.threshold(mask2gray, 240, 255, cv.THRESH_BINARY)
cv.imshow('mask2_th',mask2_th)
cv.waitKey(0)

mask2_inv = cv.bitwise_not(mask2_th)
cv.imshow('mask2_inv',mask2_inv)
cv.waitKey(0)

#img3
cv.imshow('img3',img3)
cv.waitKey(0)

__, mask3 = cv.threshold(img3, 30, 255, cv.THRESH_BINARY)
cv.imshow('mask3',mask3)
cv.waitKey(0)

mask3gray = cv.cvtColor(mask3,cv.COLOR_BGR2GRAY)
cv.imshow('mask3gray',mask3gray)
cv.waitKey(0)

__, mask3_th = cv.threshold(mask3gray, 240, 255, cv.THRESH_BINARY)
cv.imshow('mask3_th',mask3_th)
cv.waitKey(0)

mask3_inv = cv.bitwise_not(mask3_th)
cv.imshow('mask3_inv',mask3_inv)
cv.waitKey(0)

#img4
cv.imshow('img4',img4)
cv.waitKey(0)

__, mask4 = cv.threshold(img4, 30, 255, cv.THRESH_BINARY)
cv.imshow('mask4',mask4)
cv.waitKey(0)

mask4gray = cv.cvtColor(mask4,cv.COLOR_BGR2GRAY)
cv.imshow('mask4gray',mask4gray)
cv.waitKey(0)

__, mask4_th = cv.threshold(mask4gray, 240, 255, cv.THRESH_BINARY)
cv.imshow('mask4_th',mask4_th)
cv.waitKey(0)

mask4_inv = cv.bitwise_not(mask4_th)
cv.imshow('mask4_inv',mask4_inv)
cv.waitKey(0)

