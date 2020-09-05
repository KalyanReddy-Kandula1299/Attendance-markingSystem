from __future__ import division
from scipy.fftpack import fft, fftshift, dct
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
#calculate mel frequency filter bank
def mel_filterbank(N, nf, fs):
	lm =1125*np.log(1 + 300/700) 
	um =1125*np.log(1 + 4000/700)  
	mel = np.linspace(lm, um, nf+2)
	hi = 700*(np.exp(mel/1125)-1)
	fi = [int(hz * (N/2+1)/fs) for hz in hi]
	h = np.zeros((int(N/2+1),nf))
	for i in range(1,nf+1):
		for k in range(int(N/2 + 1)):
			if k < fi[i-1]:
				h[k, i-1] = 0
			elif k >= fi[i-1] and k < fi[i]:
				h[k,i-1] = (k - fi[i-1])/(fi[i] - fi[i-1])
			elif k >= fi[i] and k <= fi[i+1]:
				h[k,i-1] = (fi[i+1] - k)/(fi[i+1] - fi[i])
			else:
				h[k,i-1] = 0
	return h
def mfccf(s,fs, nfb):
	#divide into segments of 25 ms with overlap of 10ms
	nS = np.int32(0.025*fs)
	oS = np.int32(0.01*fs)
	nF = np.int32(np.ceil(len(s)/(nS-oS)))
	p = ((nS-oS)*nF) - len(s)
	if p > 0:
		X = np.append(s, np.zeros(p))
	else:
		X = s
	seg = np.zeros((nS, nF))
	t = 0
	for i in range(nF):
		seg[:,i] = X[t:t+nS]
		t = (nS-oS)*i
	#compute periodogram
	N = 512
	pg = np.zeros((nF,int(N/2 + 1)))
	for i in range(nF):
		n=np.arange(0,nS)
		w=0.5-0.5*np.cos((2*n*np.pi)/nS)
		x = seg[:,i] * w
		spect = fftshift(fft(x,N))
		pg[i,:] = abs(spect[255:])/nS
	#calculating mfccs
	fb = mel_filterbank(N, nfb, fs)
	#nfiltbank MFCCs for each frame
	mfcc = np.zeros((nfb,nF))
	for i in range(nfb):
		for k in range(nF):
			mfcc[i,k] = np.sum(pg[k,:]*fb[:,i])
	mfcc = np.log10(mfcc)
	mfcc = dct(mfcc)
	mfcc[0,:]= np.zeros(nF)
	return mfcc
def ed(d,c):
	# np.shape(d)[0] = np.shape(c)[0]
	n = np.shape(d)[1]
	p = np.shape(c)[1]
	dist= np.zeros((n,p))
	if n<p:
		for i in range(n):
			cp = np.transpose(np.tile(d[:,i], (p,1)))
			dist[i,:] = np.sum((cp - c)**2,0)
	else:
		for i in range(p):
			cp = np.transpose(np.tile(c[:,i],(n,1)))
			dist[:,i] = np.transpose(np.sum((d - cp)**2,0))
	dist = np.sqrt(dist)
	return dist
def lbg(feat, M):
	eps = 0.01
	cb = np.mean(feat, 1)
	dstr = 1
	nC = 1
	while nC < M:
		#double the size of codebook
		n_cb = np.zeros((len(cb), nC*2))
		if nC == 1:
				n_cb[:,0] = cb*(1+eps)
				n_cb[:,1] = cb*(1-eps)
		else:
			for i in range(nC):
				n_cb[:,2*i] = cb[:,i] * (1+eps)
				n_cb[:,2*i+1] = cb[:,i] * (1-eps)
		cb = n_cb
		nC = np.shape(cb)[1]
		D = ed(feat, cb)
		while np.abs(dstr) > eps:
			#nearest neighbour search
			pd = np.mean(D)
			near_cb = np.argmin(D,axis = 1)
			#cluster vectors and find new centroid
			for i in range(nC):
				#add along 3rd dimension
				cb[:,i] = np.mean(feat[:,np.where(near_cb == i)], 2).T
			#replace all NaN values with 0
			cb = np.nan_to_num(cb)
			D = ed(feat, cb)
			dstr = (pd- np.mean(D))/pd
	return cb
def training(nf):
	nSpeaker =15
	nC =16
	cb_mfcc = np.zeros((nSpeaker,nf,nC))
	fn = str()
	j=1
	for i in range(nSpeaker):
		fn = 'train' + str(i+1) + '.wav'
		(fs,s) = wavfile.read( fn)
		mfcc = mfccf(s, fs, nf)
		cb_mfcc[i,:,:] = lbg(mfcc, nC)
	print 'Training complete'
	return (cb_mfcc)
nf = 12
cb_mfcc = training(nf)
fn = str()
def mdist(feat, cb):
	i= 0
	distm = np.inf
	for k in range(np.shape(cb)[0]):
		D = ed(feat, cb[k,:,:])
		distance = np.sum(np.min(D, axis = 1))/(np.shape(D)[0])
		if distance < distm:
			distm = distance
			i = k
	return (i,distm)
(fs,s) = wavfile.read( 'unknown.wav')
mfcc = mfccf(s,fs,nf)
(sp,dm) = mdist(mfcc, cb_mfcc)
if(dm<25):
	if(sp<5):
		print "\n\n..............attendence marked for KalyanReddy............\n\n"
	elif(sp<10):
		print "\n\n...............attendence marked for ManiKanta...........\n\n"
	else:
		print "\n\n..............attendence marked for Nirosha.............\n\n"
else:
	print "\n\n......your data is not recognized,try again.......\n\n"
