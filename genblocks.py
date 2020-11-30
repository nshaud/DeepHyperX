import numpy as np
import pylab as plt
import pandas as pd
from sklearn.cluster import KMeans
import aplpy
from astropy.coordinates import SkyCoord
import astropy.units as u

def blockmask(xpc, ypc, sz, arr):
    maskblock = np.zeros(arr)
    for elem in range(xpc.size):   
        maskblock[int(ypc[elem] - sz):int(ypc[elem] + sz),
				  int(xpc[elem] - sz):int(xpc[elem] + sz)] += 1
    return maskblock

def consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)

def genblocks(mask, cat, nblocksmax=10):
	
	#########
	#Transform mask
	#########
	fig = aplpy.FITSFigure("/Users/robitaij/postdoc/Lhyrica/Taurus/taurus_L1495_250_sample_mask.fits")
	mask = np.abs(mask-2)
	
	#########
	#Read core positions
	#########
	cores = pd.read_csv(cat, sep=';', header=0)
	ra = cores['-3'].to_numpy()
	dec = cores['-4'].to_numpy()
	coords = SkyCoord(ra+' '+dec,unit=(u.hourangle, u.deg), frame='fk5')
	majaxis = cores['-32'].to_numpy()
	minaxis = cores['-33'].to_numpy()
	angles = cores['-34'].to_numpy()
	(xp, yp) = fig.world2pixel(coords.ra, coords.dec)
	
	#########
	#Clustering cores with Kmeans
	#########
	Xpos = np.vstack((xp,yp)).T
	
	Xpos = np.zeros((xp.size,2))
	Xpos[:,0] = xp
	Xpos[:,1] = yp
	kmeans = KMeans(n_clusters=nblocksmax)
	k_pred = kmeans.fit_predict(Xpos)
	
	#########
	#Estimate the size of the clusters and keep the smallest
	#########
	Sclust = np.zeros(nblocksmax)

	for nk in range(nblocksmax):
		pos = np.where(k_pred == nk)
		Sclust[nk] = np.max(np.sqrt(np.abs(xp[pos] - kmeans.cluster_centers_[nk,0])**2. + 
									np.abs(yp[pos] - kmeans.cluster_centers_[nk,1])**2.)) * 2.
	minSclust = np.min(Sclust)
	print(minSclust)
	
	sz = int(minSclust/2)
	
	#########
	#Sort core blocks
	#########
	order = np.argsort(kmeans.cluster_centers_[:,0])
	xpc = kmeans.cluster_centers_[order,0]
	ypc = kmeans.cluster_centers_[order,1]

	maskblock = blockmask(xpc, ypc, sz, mask.shape)
	
	#########
	#Update block positions
	#########
	while (np.where(maskblock == 2)[0].size !=0):
		corefill = np.zeros(xpc.size)
		flag = []
		for elem in range(xpc.size):
			#blocks outside image frame
			if ((ypc[elem] - sz) < 0):
				ypc[elem] += np.abs(ypc[elem] - sz)
			if ((ypc[elem] + sz) > mask.shape[0]):
				ypc[elem] = int(mask.shape[0]-sz)
			if ((xpc[elem] - sz) < 0):
				xpc[elem] += np.abs(xpc[elem] - sz)
			if ((xpc[elem] + sz) > mask.shape[1]):
				xpc[elem] = int(mask.shape[1]-sz)
			#remove overlaps
			canv = maskblock[int(ypc[elem] - sz):int(ypc[elem] + sz),
							 int(xpc[elem] - sz):int(xpc[elem] + sz)]
			ovl = np.where(canv == 2)
			if (ovl[0].size > 0):
				if (ovl[0].size < (sz*1.5)**2):
					xu = np.unique(ovl[1])
					yu = np.unique(ovl[0])
					splitxu = consecutive(xu)
					splityu = consecutive(yu)
					#for lapse in range(len(splitxu)):
					lapse = 0
					if (splitxu[lapse].size < splityu[lapse].size) and (splitxu[lapse][0] < sz):
						xpc[elem] += (np.max(splitxu[lapse])+1)
					if (splityu[lapse].size < splitxu[lapse].size) and (splityu[lapse][0] < sz):
						ypc[elem] += (np.max(splityu[lapse])+1)
					maskblock = blockmask(xpc, ypc, sz, mask.shape)
					nz = np.where(mask[int(ypc[elem] - sz):int(ypc[elem] + sz)
									   ,int(xpc[elem] - sz):int(xpc[elem] + sz)] == 0)[0].size
					corefill[elem] = float(nz) / (float(sz)*2.)**2
					#canv = maskblock[int(ypc[elem] - sz):int(ypc[elem] + sz),int(xpc[elem] - sz):int(xpc[elem] + sz)]
					#ovl = np.where(canv == 2)
					#if (ovl[0].size > (sz*1.5)**2):
					#    flag.append(elem)
				else:
					flag.append(elem)
					nz = np.where(mask[int(ypc[elem] - sz):int(ypc[elem] + sz),
									   int(xpc[elem] - sz):int(xpc[elem] + sz)] == 0)[0].size
					corefill[elem] = float(nz) / (float(sz)*2.)**2
			else:
				nz = np.where(mask[int(ypc[elem] - sz):int(ypc[elem] + sz),
								   int(xpc[elem] - sz):int(xpc[elem] + sz)] == 0)[0].size
				corefill[elem] = float(nz) / (float(sz)*2.)**2

		#Filtering low core fill factor        
		filt = np.where((corefill-np.mean(corefill))/np.std(corefill) < -1.4)[0]

		if filt.size > 0:
			for elem in filt:
				flag.append(elem)

		flag = np.unique(flag).tolist()
		xpc = np.delete(xpc,flag)
		ypc = np.delete(ypc,flag)

		maskblock = blockmask(xpc, ypc, sz, mask.shape)
		
	#########
	#Background blocks
	#########
	
	xBclist = []
	yBclist = []

	xblock = np.int(mask.shape[1] / (sz*2))
	yblock = np.int(mask.shape[0] / (sz*2))

	for yl in range(yblock):
		for xl in range(xblock):
			BGblock = maskblock[yl*sz*2:yl*sz*2+sz*2,xl*sz*2:xl*sz*2+sz*2]
			if (np.where(BGblock == 1)[0].size == 0):
				xBclist.append(xl*sz*2+sz)
				yBclist.append(yl*sz*2+sz)

	xBc = np.array(xBclist)
	yBc = np.array(yBclist)
	
	plt.tight_layout()
	
	return xpc, ypc, xBc, yBc, sz