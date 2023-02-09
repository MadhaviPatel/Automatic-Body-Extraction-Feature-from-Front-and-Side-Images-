import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import imutils

selectedPoints=[]		#selected points for euclidian distance
ppm=None 				#pixel per metric ratio

def midpoint(pA, pB):
	return ((pA[0] + pB[0]) * 0.5, (pA[1] + pB[1]) * 0.5)

def calculatePPM(cnt):				#calculation of pixel per metric ratio
	box = cv2.minAreaRect(cnt)
	box = cv2.boxPoints(box)
	box = np.array(box, dtype="int")
	box = perspective.order_points(box)
	(l, r, br, bl) = box
	(tlrX, tlrY) = midpoint(l, r)
	(blrX, blrY) = midpoint(bl, br)
	(tblX, tblY) = midpoint(l, bl)
	(tbrX, tbrY) = midpoint(r, br)
	distA = dist.euclidean((tlrX, tlrY), (blrX, blrY))
	distB = dist.euclidean((tblX, tblY), (tbrX, tbrY))
	pixelsPerMetric = distA / 20
	return pixelsPerMetric
def onclick(event):				#Capturing on click event
    if event.xdata != None and event.ydata != None:
    	selectedPoints.append((event.xdata, event.ydata))
    	if len(selectedPoints)==2 and ppm!=None :
    		dA = dist.euclidean(selectedPoints[0], selectedPoints[1])
    		print(selectedPoints)
    		print('Real World Dimension is :- ',dA/ppm)
    		selectedPoints.pop()
    		selectedPoints.pop()

def featurePointDetection(image) :			#method for detecting feature points
	(startX,startY) = (0,0)
	p = []
	p_xy = []
	v = []
	v_xy = []
	f_xy = []
	(height,width) = image.shape[:2]
	for i in range(0,height):					#to select the first most pixel
		for j in range(0,width):
			if image[i,j] !=0:
				print('[',i,',',j,']')
				startX = i
				startY = j
				break;
		if startX != 0 or startY != 0 :
			break;
	p.append(0)
	p_xy.append((startX,startY))
	tempX = startX
	tempY = startY
	count=0
	while image[tempX,tempY] !=0 and count<50:							#free mans chain code algorithm
		if image[tempX,tempY+1] !=0 and not ((tempX,tempY+1) in p_xy):
			image[tempX,tempY]=100
			p.append(0)
			tempX = tempX
			tempY = tempY+1
			p_xy.append((tempX,tempY))
			#print('[',tempX,',',tempY,']')
		elif image[tempX-1,tempY+1] !=0 and not ((tempX-1,tempY+1) in p_xy):
			image[tempX,tempY]=100
			p.append(1)
			tempX = tempX-1
			tempY = tempY+1
			p_xy.append((tempX,tempY))
			#print('[',tempX,',',tempY,']')
		elif image[tempX-1,tempY] !=0 and not ((tempX-1,tempY) in p_xy):
			image[tempX,tempY]=100
			p.append(2)
			tempX = tempX-1
			tempY = tempY
			p_xy.append((tempX,tempY))
			#print('[',tempX,',',tempY,']')
		elif image[tempX-1,tempY-1] !=0 and not ((tempX-1,tempY-1) in p_xy):
			image[tempX,tempY]=100
			p.append(3)
			tempX = tempX-1
			tempY = tempY-1
			p_xy.append((tempX,tempY))
			inc=1
			#print('[',tempX,',',tempY,']')
		elif image[tempX,tempY-1] !=0and not ((tempX,tempY-1) in p_xy):
			image[tempX,tempY]=100
			p.append(4)
			tempX = tempX
			tempY = tempY-1
			p_xy.append((tempX,tempY))
			#print('[',tempX,',',tempY,']')
		elif image[tempX+1,tempY-1] !=0 and not ((tempX+1,tempY-1) in p_xy):
			image[tempX,tempY]=100
			p.append(5)
			tempX = tempX+1
			tempY = tempY-1
			p_xy.append((tempX,tempY))
			#print('[',tempX,',',tempY,']')
		elif image[tempX+1,tempY] !=0and not ((tempX+1,tempY) in p_xy):
			image[tempX,tempY]=100
			p.append(6)
			tempX = tempX+1
			tempY = tempY
			p_xy.append((tempX,tempY))
			#print('[',tempX,',',tempY,']')
		elif image[tempX+1,tempY+1] !=0 and not ((tempX+1,tempY+1) in p_xy):
			image[tempX,tempY]=100
			p.append(7)
			tempX = tempX+1
			tempY = tempY+1
			p_xy.append((tempX,tempY))
			#print('[',tempX,',',tempY,']')	
		else :
			image[p_xy[-1]]=0
			p_xy.pop()
			p.pop()
			tempX,tempY=p_xy[-1]
			count+=1
	#print(p)
	#print(len(p),' and ',len(p_xy)) 
	v.append(p[0])
	v_xy.append(p_xy[0])
	for i in range(1,len(p)):			
		if p[i]!=p[i-1]:
			v.append(p[i])
			v_xy.append(p_xy[i])
	#print(len(v),' and ',len(v_xy))
	f_xy.append(v_xy[0])
	for i in range(1,len(v)-1,5):					#detecting feature points based on conditions
		if v[i]-v[i-1]==-1 and v[i+1]-v[i]==1 :
			f_xy.append(v_xy[i])
		elif v[i]-v[i-1]==-1 and v[i+1]-v[i]==-7:
			f_xy.append(v_xy[i])
		elif v[i]-v[i-1]==-7 and v[i+1]-v[i]==-1:
			f_xy.append(v_xy[i])
		elif v[i]-v[i-1]==1 and v[i+1]-v[i]==7:
			f_xy.append(v_xy[i])
		elif v[i]-v[i-1]==7 and v[i+1]-v[i]==1:
			f_xy.append(v_xy[i])
		elif abs(v[i]-v[i-1])==2:
			f_xy.append(v_xy[i])
	print('Number of Feature points detected :- ',len(f_xy))
	ax = plt.gca()
	fig = plt.gcf()
	implot = ax.imshow(image,cmap='gray')
	plt.plot([i[1] for i in f_xy],[j[0] for j in f_xy],'r+') 
	cid = fig.canvas.mpl_connect('button_press_event', onclick)
	print('Please select any two feature points')
	plt.show()
frontImage = cv2.imread('Test3.png',cv2.IMREAD_GRAYSCALE)  				#pre-processing
cv2.imshow('Gray-FrontImage',frontImage)
cv2.waitKey(0)
sideImage = cv2.imread('Test4.png',cv2.IMREAD_GRAYSCALE)
cv2.imshow('Gray-SideImage',sideImage)
cv2.waitKey(0)
gaussFrontImage = cv2.GaussianBlur(frontImage,(3,3),0) 
cv2.imshow('Gaussian-FrontImage',gaussFrontImage)
cv2.waitKey(0)
gaussSideImage = cv2.GaussianBlur(sideImage,(3,3),0) 
cv2.imshow('Gaussian-SideImage',gaussSideImage)
cv2.waitKey(0)
th,binaryFrontImage = cv2.threshold(gaussFrontImage,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) 
cv2.imshow('Binary-FrontImage',binaryFrontImage)
cv2.waitKey(0)
th,binarySideImage = cv2.threshold(gaussSideImage,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) 
cv2.imshow('Binary-SideImage',binarySideImage)
cv2.waitKey(0)
vf=np.median(binaryFrontImage)
sigmaf=0.33
lowerf=int(max(0,(1.0-sigmaf)*vf))
higherf=int(min(255,(1.0+sigmaf)*vf))
cannyFrontImage = cv2.Canny(binaryFrontImage,lowerf,higherf)
cv2.imshow('CannyEdge-FrontImage',cannyFrontImage)
cv2.waitKey(0)
vs=np.median(binarySideImage)
sigmas=0.33
lowers=int(max(0,(1.0-sigmas)*vs))
highers=int(min(255,(1.0+sigmas)*vs))
cannySideImage = cv2.Canny(binarySideImage,lowers,highers)
cv2.imshow('CannyEdge-SideImage',cannySideImage)
cv2.waitKey(0)
kernelSizes = [(3,3)]
for kernelSize in kernelSizes:
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)
	cannyFrontImage = cv2.morphologyEx(cannyFrontImage, cv2.MORPH_CLOSE, kernel)
	cannySideImage = cv2.morphologyEx(cannySideImage, cv2.MORPH_CLOSE, kernel)
cv2.imshow('Morphology-FrontImage',cannyFrontImage)
cv2.waitKey(0)
cv2.imshow('Morphology-SideImage',cannySideImage)
cv2.waitKey(0)
contoursf = cv2.findContours(cannyFrontImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contoursf = imutils.grab_contours(contoursf)
(contoursf, _) = contours.sort_contours(contoursf)
ppm=calculatePPM(contoursf[0])
#print('ppm is :- ',ppmf)
resultf = np.zeros_like(cannyFrontImage)
cv2.drawContours(resultf, contoursf[1], -1, (255,255,255), 1)
cv2.imshow('ClosedContour-FrontImage',resultf)
cv2.waitKey(0)
featurePointDetection(resultf)
contourss = cv2.findContours(cannySideImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contourss = imutils.grab_contours(contourss)
(contourss, _) = contours.sort_contours(contourss)
ppm=calculatePPM(contourss[0])
#print('ppm is :- ',ppms)
results = np.zeros_like(cannySideImage)
cv2.drawContours(results, contourss[1], -1, (255,255,255), 1)
cv2.imshow('ClosedContour-SideImage',results)
cv2.waitKey(0)
featurePointDetection(results)