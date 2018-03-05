import numpy as np
import cv2
from matplotlib import pyplot as plt
from lis import lis



def kp_verify():

	query_img = cv2.imread('/home/goutham/cv/opencv/opencv-3.1.0/samples/data/box.png',0)          # queryImage
	search_img = cv2.imread('/home/goutham/cv/opencv/opencv-3.1.0/samples/data/box_in_scene.png',0) # trainImage
	query_ang = [0, 120, 240]
	search_ang = [50, 100, 150, 200, 250, 300, 350]
	print('Rotating query_img')
	for ang in query_ang:
		query_img_t = rot(query_img, ang)
		print(ang, find_lis(query_img_t, search_img))

	print('Rotating search_img')
	for ang in search_ang:
		search_img_t = rot(search_img, ang)
		print(ang, find_lis(query_img, search_img_t))

def rot(img, ang):

	rows,cols = img.shape
	tx = (cols/2); ty = (rows/2)
	M = cv2.getRotationMatrix2D((cols/2,rows/2),ang,1)
	tr = np.array([[0, 0, tx],[0, 0, ty]])
	M = M + tr
	dst = cv2.warpAffine(img,M,(cols*2,rows*2)) 
	return dst


def find_lis(img1, img2):


	# Initiate SIFT detector
	orb = cv2.ORB_create()

	# find the keypoints and descriptors with SIFT
	kp1, des1 = orb.detectAndCompute(img1,None)
	kp2, des2 = orb.detectAndCompute(img2,None)

	# create BFMatcher object
	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
	
	# Match descriptors.
	matches = bf.match(des1,des2)

	# Sort them in the order of their distance.
	matches = sorted(matches, key = lambda x:x.distance)
	# print('len:',len(matches))

	img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],None, flags=2)

	# cv2.imshow('1', img3)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	dict_train_to_query = {}
	dict_query_to_train = {}
	query_idx = []

	for x in matches:
		query_idx.append(x.queryIdx)
		dict_train_to_query[x.trainIdx] = x.queryIdx
		dict_query_to_train[x.queryIdx] = x.trainIdx

	train_idx = dict_train_to_query.keys()
	sorted_query_kps = []


	for x in query_idx:
		sorted_query_kps.append((kp1[x].pt[0], x, dict_query_to_train[x]))

	sorted_query_kps = sorted(sorted_query_kps, key=lambda x:x[0])

	wlis_query = {}

	for i, x in enumerate(sorted_query_kps):
		wlis_query[x[1]] = i

	sorted_train_kps = []
	for x in train_idx:
		sorted_train_kps.append((kp2[x].pt[0], x))

	sorted_train_kps = sorted(sorted_train_kps, key=lambda x:x[0])

	wlis_train = []

	for x in sorted_train_kps:
		q_idx = dict_train_to_query[x[1]]
		idx = wlis_query[q_idx]
		wlis_train.append(idx)

	
	return lis(wlis_train)


kp_verify()
