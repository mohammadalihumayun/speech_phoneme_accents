#
import scipy.io.wavfile
import numpy as np
#import winsound
import scipy.signal
import matplotlib.pyplot as plt
import librosa.feature
from sklearn.preprocessing import normalize
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Input, Dense,PReLU, ReLU
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import  Conv1D, Flatten, MaxPooling1D, concatenate, AveragePooling1D, GlobalAvgPool1D
import csv
#from spafe.features.gfcc import gfcc as spf_gfc
#from scipy.stats import mode
from sklearn.preprocessing import normalize
from sklearn.preprocessing import OneHotEncoder as ohe
import gc
import copy 
import statistics
from scipy import signal
#import dask.array as da
from sklearn import svm
from scipy import stats
from scipy.spatial.distance import cdist

#

def spkphnsets():
 sumdesc=np.concatenate((np.load(infpath+'summarydescription.npy'),np.load(infpath+'sent_desc.npy').reshape(-1,1)),axis=1)
 native=np.unique(sumdesc[:,0])
 phonemes=np.unique(sumdesc[:,2])
 spkrs=np.unique(sumdesc[:,3],return_index=True)[0]
 spkix=np.unique(sumdesc[:,3],return_index=True)[1]
 spklg=sumdesc[:,0][spkix]
 spkgd=[]
 for st in sumdesc[:,4][spkix]:
     spkgd.append(st[13])   
 spkgd=np.array(spkgd)
 mspkrs=[]
 fspkrs=[]
 for lg in native:
     mspkrs.append(spkrs[(spkgd=='m')&(spklg==lg)][:tms])
     fspkrs.append(spkrs[(spkgd=='f')&(spklg==lg)][:tfs])
 np.random.seed(seed=sd)
 [np.random.shuffle(x) for x in mspkrs]
 [np.random.shuffle(x) for x in fspkrs]
 tmspkrs=np.array(mspkrs)[:,:int(tms*tpct)].flatten()
 tfspkrs=np.array(fspkrs)[:,:int(tfs*tpct)].flatten()
 cmspkrs=np.array(mspkrs)[:,int(tms*tpct):].flatten()
 cfspkrs=np.array(fspkrs)[:,int(tfs*tpct):].flatten()
 cmsents=np.unique(sumdesc[:,4][np.in1d(sumdesc[:,3],cmspkrs)])
 cfsents=np.unique(sumdesc[:,4][np.in1d(sumdesc[:,3],cfspkrs)])
 cmsentswav=[x+'.wav' for x in cmsents]
 cfsentswav=[x+'.wav' for x in cfsents]
 del sumdesc
 gc.collect()
 return native,phonemes,tmspkrs,tfspkrs,cmspkrs,cfspkrs,cmsents,cfsents,cmsentswav,cfsentswav

#
def phcnnprediction():
 pt=1
 inputdim1=feats.shape
 outputdim1=catlabels.shape[1]
 input_flat1 = Input(shape=(inputdim1[1:]))
 h=Conv1D(15, kernel_size=16,strides=3, padding='same')(input_flat1)#
 h=BatchNormalization()(h)
 h=MaxPooling1D()(h)# 
 h=Flatten()(h) 
 output_layer1 = Dense(outputdim1, activation='softmax')(h)
 emodel1 = Model(input_flat1,output_layer1)
 emodel1.compile(optimizer='Adamax', loss='categorical_crossentropy')
 emodel1.save_weights('model.h5')
 emodel1.summary()
 #es = EarlyStopping(monitor='val_loss', mode='min',verbose=1,min_delta=0, patience=pt)
 es = EarlyStopping(monitor='loss', mode='min',verbose=1,min_delta=0, patience=pt)
 #hist= emodel1.fit(tfp[:,:featlen,:],tlp,epochs=10,batch_size=32,shuffle=True,validation_data=(cfp[:,:featlen,:], clp),callbacks=[es],verbose=1) 
 mphout=np.zeros((len(phonemes),len(cmspkrs),len(native)))
 fphout=np.zeros((len(phonemes),len(cfspkrs),len(native)))
 mphoutsents=np.zeros((len(mfcc_phone_8_16),len(cmsents),len(native)))
 fphoutsents=np.zeros((len(phonemes),len(cfsents),len(native)))
 phi=0
 for ph in phonemes:
     tbidx=(np.in1d(sumdesc[:,3],np.concatenate((tmspkrs,tfspkrs),axis=0))&(sumdesc[:,2]==ph))
     tmidx=(np.in1d(sumdesc[:,3],tmspkrs))&(sumdesc[:,2]==ph)
     tfidx=(np.in1d(sumdesc[:,3],tfspkrs))&(sumdesc[:,2]==ph)
     cmidx=(np.in1d(sumdesc[:,3],cmspkrs))&(sumdesc[:,2]==ph)
     cfidx=(np.in1d(sumdesc[:,3],cfspkrs))&(sumdesc[:,2]==ph)
     if ((testdegraded=='filter')and (augmentation=='yes')):
         cfidx[:int(len(sumdesc)/2)]=np.repeat(False,int(len(sumdesc)/2))
         cmidx[:int(len(sumdesc)/2)]=np.repeat(False,int(len(sumdesc)/2))
     elif ((testdegraded=='clean')and (augmentation=='yes')):
         cfidx[int(len(sumdesc)/2):]=np.repeat(False,int(len(sumdesc)/2))
         cmidx[int(len(sumdesc)/2):]=np.repeat(False,int(len(sumdesc)/2))
     elif (testdegraded=='filter_onlytest')& (augmentation=='yes'):
         cfidx[:int(len(sumdesc)/2)]=np.repeat(False,int(len(sumdesc)/2))
         cmidx[:int(len(sumdesc)/2)]=np.repeat(False,int(len(sumdesc)/2))
         tfidx[int(len(sumdesc)/2):]=np.repeat(False,int(len(sumdesc)/2))
         tmidx[int(len(sumdesc)/2):]=np.repeat(False,int(len(sumdesc)/2))
   
     if gendertrain=='yes':
      print('Training_Male_Phone',ph)
      emodel1.load_weights('model.h5')
      histm= emodel1.fit(feats[tmidx],catlabels[tmidx],epochs=100,batch_size=32,shuffle=True,callbacks=[es],verbose=1) 
      msvmprob = emodel1.predict(feats[cmidx])
      print('FFN_Training_Female_Phone',ph)
      emodel1.load_weights('model.h5')
      histf= emodel1.fit(feats[tfidx],catlabels[tfidx],epochs=100,batch_size=32,shuffle=True,callbacks=[es],verbose=1) 
      fsvmprob = emodel1.predict(feats[cfidx])
     else:
      print('Training_Joint-gender_Phone',ph)
      emodel1.load_weights('model.h5')
      hist= emodel1.fit(feats[tbidx],catlabels[tbidx],epochs=100,batch_size=32,shuffle=True,callbacks=[es],verbose=1) 
      msvmprob = emodel1.predict(feats[cmidx])
      fsvmprob = emodel1.predict(feats[cfidx])
     spj=0
     for sp in cmspkrs:
         mphout[phi][spj]=(np.mean(msvmprob[(sumdesc[:,3][cmidx]==sp)],axis=0))
         spj=spj+1
     spj=0
     for sp in cfspkrs:
         fphout[phi][spj]=(np.mean(fsvmprob[(sumdesc[:,3][cfidx]==sp)],axis=0))
         spj=spj+1
     spj=0
     for sp in cmsents:
         mphoutsents[phi][spj]=(np.mean(msvmprob[(sumdesc[:,4][cmidx]==sp)],axis=0))
         spj=spj+1
     spj=0
     for sp in cfsents:
         fphoutsents[phi][spj]=(np.mean(fsvmprob[(sumdesc[:,4][cfidx]==sp)],axis=0))
         spj=spj+1
     phi=phi+1
 mphout[np.isnan(mphout)]=0
 fphout[np.isnan(fphout)]=0
 mphoutsents[np.isnan(mphoutsents)]=0
 fphoutsents[np.isnan(fphoutsents)]=0
 # wtd
 if wtdphn=='yes':
     mphspkout=np.average(mphout,axis=0, weights=mpwts_spkr)
     fphspkout=np.average(fphout,axis=0, weights=fpwts_spkr)
     mphsntout=np.average(mphoutsents,axis=0, weights=mpwts)
     fphsntout=np.average(fphoutsents,axis=0, weights=fpwts)
 else:
     mphspkout=np.mean(mphout,axis=0)
     fphspkout=np.mean(fphout,axis=0)
     mphsntout=np.mean(mphoutsents,axis=0)
     fphsntout=np.mean(fphoutsents,axis=0)
 #phone accuracy
 gc.collect()
 if calcphacc=='yes':
  mspix=[sumdesc[:,3].tolist().index(j) for j in cmspkrs]
  fspix=[sumdesc[:,3].tolist().index(j) for j in cfspkrs]
  mstix=[np.where(j==sumdesc[:,4])[0][0] for j in cmsents]
  fstix=[np.where(j==sumdesc[:,4])[0][0] for j in cfsents]
  len(mstix),len(fstix), len(mspix),len(fspix)
  lbl=np.argmax(catlabels,axis=1)
  accargs=[[mphout,mspix,'male-spkr'],[fphout,fspix,'fema-spkr'],[mphoutsents,mstix,'male-sent'],[fphoutsents,fstix,'fema-sent']]
  print('writing_phone-acc-csv...')
  for phacc in accargs:
     for phnm in range(len(phonemes)):
         mskzro=np.max(phacc[0][phnm],axis=1)!=0
         #print(mskzro)
         #print(np.max(phacc[0][phnm],axis=1))
         with open(phncsv, 'a', newline='') as csvfile:
            res = csv.writer(csvfile)
            res.writerow([sd,phacc[2],phonemes[phnm],sum(np.array(np.argmax(phacc[0][phnm],axis=1))[mskzro]==np.array(lbl[phacc[1]])[mskzro])/len(np.array(phacc[1])[mskzro]),len(phacc[1]),np.array(np.argmax(phacc[0][phnm],axis=1))[mskzro],np.array(lbl[phacc[1]])[mskzro],np.max(phacc[0][phnm],axis=1)])
 return mphspkout,fphspkout,mphsntout,fphsntout

# 
def ltcnnprediction():
 ## train test
 tbidx=(np.in1d(sumdesc[:,3],np.concatenate((tmspkrs,tfspkrs),axis=0)))
 tmidx=(np.in1d(sumdesc[:,3],tmspkrs))
 tfidx=(np.in1d(sumdesc[:,3],tfspkrs))
 cbidx=(np.in1d(sumdesc[:,3],np.concatenate((cmspkrs,cfspkrs),axis=0)))
 cmidx=(np.in1d(sumdesc[:,4],cmsentswav))
 cfidx=(np.in1d(sumdesc[:,4],cfsentswav))
 if (testdegraded=='filter')& (augmentation=='yes'):
    cfidx[:int(len(sumdesc)/2)]=np.repeat(False,int(len(sumdesc)/2))
    cmidx[:int(len(sumdesc)/2)]=np.repeat(False,int(len(sumdesc)/2))
    cbidx[:int(len(sumdesc)/2)]=np.repeat(False,int(len(sumdesc)/2))
 elif (testdegraded=='clean')& (augmentation=='yes'):
    cfidx[int(len(sumdesc)/2):]=np.repeat(False,int(len(sumdesc)/2))
    cmidx[int(len(sumdesc)/2):]=np.repeat(False,int(len(sumdesc)/2))
    cbidx[int(len(sumdesc)/2):]=np.repeat(False,int(len(sumdesc)/2))
 elif (testdegraded=='filter_onlytest')& (augmentation=='yes'):
    cfidx[:int(len(sumdesc)/2)]=np.repeat(False,int(len(sumdesc)/2))
    cmidx[:int(len(sumdesc)/2)]=np.repeat(False,int(len(sumdesc)/2))
    cbidx[:int(len(sumdesc)/2)]=np.repeat(False,int(len(sumdesc)/2))
    tfidx[int(len(sumdesc)/2):]=np.repeat(False,int(len(sumdesc)/2))
    tmidx[int(len(sumdesc)/2):]=np.repeat(False,int(len(sumdesc)/2))
    tbidx[int(len(sumdesc)/2):]=np.repeat(False,int(len(sumdesc)/2))
    
 gc.collect()
 featlen=int(feats.shape[1]/ftlenratio)#750#1000#1500
 #''' cnn training + prediction
 indepcnnparams=[64,32,8]
 mpl=int(indepcnnparams[0]/1)
 krl=int(indepcnnparams[1]/1)
 std=int(indepcnnparams[2]/1)
 pt=1
 inputdim1=feats[:,:featlen,:].shape
 outputdim1=catlabels.shape[1]
 input_flat1 = Input(shape=(inputdim1[1:]))
 h=Conv1D(10, kernel_size=krl,strides=std, padding='same')(input_flat1)#
 h=BatchNormalization()(h)
 h=MaxPooling1D(mpl)(h)# 
 h=PReLU()(h) 
 h=Flatten()(h) 
 output_layer1 = Dense(outputdim1, activation='softmax')(h)
 emodel1 = Model(input_flat1,output_layer1)
 emodel1.compile(optimizer='Adamax', loss='categorical_crossentropy')
 emodel1.save_weights('cnnmodel.h5')
 emodel1.summary()
 #es = EarlyStopping(monitor='val_loss', mode='min',verbose=1,min_delta=0, patience=pt)
 es = EarlyStopping(monitor='loss', mode='min',verbose=1,min_delta=0, patience=pt)
 if gendertraincnn=='yes':
  emodel1.load_weights('cnnmodel.h5')
  print('Training_male')
  histm= emodel1.fit(feats[tmidx][:,:featlen,:],catlabels[tmidx],epochs=100,batch_size=32,shuffle=True,callbacks=[es],verbose=1) 
  mcnout = emodel1.predict(feats[cmidx][:,:featlen,:])
  emodel1.load_weights('cnnmodel.h5')
  print('Training_female')
  histf= emodel1.fit(feats[tfidx][:,:featlen,:],catlabels[tfidx],epochs=100,batch_size=32,shuffle=True,callbacks=[es],verbose=1) 
  fcnout = emodel1.predict(feats[cfidx][:,:featlen,:])
 else:
  emodel1.load_weights('cnnmodel.h5')
  print('Training_joint-gender')
  hist= emodel1.fit(feats[tbidx][:,:featlen,:],catlabels[tbidx],epochs=100,batch_size=32,shuffle=True,callbacks=[es],verbose=1) 
  mcnout = emodel1.predict(feats[cmidx][:,:featlen,:])
  fcnout = emodel1.predict(feats[cfidx][:,:featlen,:])
 mcnsntout=[]
 fcnsntout=[]
 for st in cmsentswav:
     mcnsntout.append(np.mean(mcnout[(sumdesc[:,4][cmidx]==st)],axis=0))
 for st in cfsentswav:
     fcnsntout.append(np.mean(fcnout[(sumdesc[:,4][cfidx]==st)],axis=0))
 mcnspkout=[]
 fcnspkout=[]
 for sp in cmspkrs:
     mcnspkout.append(np.mean(mcnout[(sumdesc[:,3][cmidx]==sp)],axis=0))
 for sp in cfspkrs:
     fcnspkout.append(np.mean(fcnout[(sumdesc[:,3][cfidx]==sp)],axis=0))
 return np.array(mcnspkout),np.array(fcnspkout),np.array(mcnsntout),np.array(fcnsntout)
 #'''###


def ffnprediction():
 pt=1
 inputdim1=feats.shape
 outputdim1=catlabels.shape[1]
 input_flat1 = Input(shape=(inputdim1[1:]))
 h=Dense(7)(input_flat1)#
 h=BatchNormalization()(h)
 #h=PReLU()(h) 
 output_layer1 = Dense(outputdim1, activation='softmax')(h)
 emodel1 = Model(input_flat1,output_layer1)
 emodel1.compile(optimizer='Adamax', loss='categorical_crossentropy')
 emodel1.save_weights('model.h5')
 emodel1.summary()
 #es = EarlyStopping(monitor='val_loss', mode='min',verbose=1,min_delta=0, patience=pt)
 es = EarlyStopping(monitor='loss', mode='min',verbose=1,min_delta=0, patience=pt)
 #hist= emodel1.fit(tfp[:,:featlen,:],tlp,epochs=10,batch_size=32,shuffle=True,validation_data=(cfp[:,:featlen,:], clp),callbacks=[es],verbose=1) 
 mphout=np.zeros((len(phonemes),len(cmspkrs),len(native)))
 fphout=np.zeros((len(phonemes),len(cfspkrs),len(native)))
 mphoutsents=np.zeros((len(phonemes),len(cmsents),len(native)))
 fphoutsents=np.zeros((len(phonemes),len(cfsents),len(native)))
 phi=0
 for ph in phonemes:
     tbidx=(np.in1d(sumdesc[:,3],np.concatenate((tmspkrs,tfspkrs),axis=0))&(sumdesc[:,2]==ph))
     tmidx=(np.in1d(sumdesc[:,3],tmspkrs))&(sumdesc[:,2]==ph)
     tfidx=(np.in1d(sumdesc[:,3],tfspkrs))&(sumdesc[:,2]==ph)
     cmidx=(np.in1d(sumdesc[:,3],cmspkrs))&(sumdesc[:,2]==ph)
     cfidx=(np.in1d(sumdesc[:,3],cfspkrs))&(sumdesc[:,2]==ph)
     if gendertrain=='yes':
      print('FFN_Training_Male_Phone',ph)
      emodel1.load_weights('model.h5')
      histm= emodel1.fit(feats[tmidx],catlabels[tmidx],epochs=100,batch_size=32,shuffle=True,callbacks=[es],verbose=1) 
      msvmprob = emodel1.predict(feats[cmidx])
      print('FFN_Training_Female_Phone',ph)
      emodel1.load_weights('model.h5')
      histf= emodel1.fit(feats[tfidx],catlabels[tfidx],epochs=100,batch_size=32,shuffle=True,callbacks=[es],verbose=1) 
      fsvmprob = emodel1.predict(feats[cfidx])
     else:
      print('FFN_Training_Joint-gender_Phone',ph)
      emodel1.load_weights('model.h5')
      hist= emodel1.fit(feats[tbidx],catlabels[tbidx],epochs=50,batch_size=32,shuffle=True,callbacks=[es],verbose=1) 
      msvmprob = emodel1.predict(feats[cmidx])
      fsvmprob = emodel1.predict(feats[cfidx])
     spj=0
     for sp in cmspkrs:
         mphout[phi][spj]=(np.mean(msvmprob[(sumdesc[:,3][cmidx]==sp)],axis=0))
         spj=spj+1
     spj=0
     for sp in cfspkrs:
         fphout[phi][spj]=(np.mean(fsvmprob[(sumdesc[:,3][cfidx]==sp)],axis=0))
         spj=spj+1
     spj=0
     for sp in cmsents:
         mphoutsents[phi][spj]=(np.mean(msvmprob[(sumdesc[:,4][cmidx]==sp)],axis=0))
         spj=spj+1
     spj=0
     for sp in cfsents:
         fphoutsents[phi][spj]=(np.mean(fsvmprob[(sumdesc[:,4][cfidx]==sp)],axis=0))
         spj=spj+1
     phi=phi+1
 mphout[np.isnan(mphout)]=0
 fphout[np.isnan(fphout)]=0
 mphoutsents[np.isnan(mphoutsents)]=0
 fphoutsents[np.isnan(fphoutsents)]=0
 if wtdphn=='yes':
     mphspkout=np.average(mphout,axis=0, weights=mpwts_spkr)
     fphspkout=np.average(fphout,axis=0, weights=fpwts_spkr)
     mphsntout=np.average(mphoutsents,axis=0, weights=mpwts)
     fphsntout=np.average(fphoutsents,axis=0, weights=fpwts)
 else:
     mphspkout=np.mean(mphout,axis=0)
     fphspkout=np.mean(fphout,axis=0)
     mphsntout=np.mean(mphoutsents,axis=0)
     fphsntout=np.mean(fphoutsents,axis=0)
 # phone accuracy
 if calcphacc=='yes':
  mspix=[sumdesc[:,3].tolist().index(j) for j in cmspkrs]
  fspix=[sumdesc[:,3].tolist().index(j) for j in cfspkrs]
  mstix=[np.where(j==sumdesc[:,4])[0][0] for j in cmsents]
  fstix=[np.where(j==sumdesc[:,4])[0][0] for j in cfsents]
  len(mstix),len(fstix), len(mspix),len(fspix)
  lbl=np.argmax(catlabels,axis=1)
  accargs=[[mphout,mspix,'male-spkr'],[fphout,fspix,'fema-spkr'],[mphoutsents,mstix,'male-sent'],[fphoutsents,fstix,'fema-sent']]
  print('writing_phone-acc-csv...')
  for phacc in accargs:
     for phnm in range(len(phonemes)):    
         mskzro=np.max(phacc[0][phnm],axis=1)!=0
         with open('phone-acc.csv', 'a', newline='') as csvfile:
            res = csv.writer(csvfile)
            res.writerow([sd,phacc[2],phonemes[phnm],sum(np.array(np.argmax(phacc[0][phnm],axis=1))[mskzro]==np.array(lbl[phacc[1]])[mskzro])/len(np.array(phacc[1])[mskzro]),len(phacc[1]),np.array(np.argmax(phacc[0][phnm],axis=1))[mskzro],np.array(lbl[phacc[1]])[mskzro],np.max(phacc[0][phnm],axis=1)])
 return mphspkout,fphspkout,mphsntout,fphsntout

# %% 
def final_acc_dual():
 mspix=[sumdesc[:,3].tolist().index(j) for j in cmspkrs]
 fspix=[sumdesc[:,3].tolist().index(j) for j in cfspkrs]
 tlensp=len(mspix)+len(fspix)
 mstix=[sumdesc[:,4].tolist().index(j) for j in cmsentswav]
 fstix=[sumdesc[:,4].tolist().index(j) for j in cfsentswav]
 tlenst=len(mstix)+len(fstix)
 
 msp=sum(np.argmax((j*mphspkoutmfc+l*mcnspkout),axis=1)==np.argmax(catlabels,axis=1)[mspix])
 fsp=sum(np.argmax((j*fphspkoutmfc+l*fcnspkout),axis=1)==np.argmax(catlabels,axis=1)[fspix])
 mst=sum(np.argmax((j*mphsntoutmfc+l*mcnsntout),axis=1)==np.argmax(catlabels,axis=1)[mstix])
 fst=sum(np.argmax((j*fphsntoutmfc+l*fcnsntout),axis=1)==np.argmax(catlabels,axis=1)[fstix])
 
 amp=sum(np.argmax(mphspkoutmfc,axis=1)==np.argmax(catlabels,axis=1)[mspix])
 afp=sum(np.argmax(fphspkoutmfc,axis=1)==np.argmax(catlabels,axis=1)[fspix])
 cmp=sum(np.argmax(mcnspkout,axis=1)==np.argmax(catlabels,axis=1)[mspix])
 cfp=sum(np.argmax(fcnspkout,axis=1)==np.argmax(catlabels,axis=1)[fspix]) 
 amt=sum(np.argmax(mphsntoutmfc,axis=1)==np.argmax(catlabels,axis=1)[mstix])
 aft=sum(np.argmax(fphsntoutmfc,axis=1)==np.argmax(catlabels,axis=1)[fstix])
 cmt=sum(np.argmax(mcnsntout,axis=1)==np.argmax(catlabels,axis=1)[mstix])
 cft=sum(np.argmax(fcnsntout,axis=1)==np.argmax(catlabels,axis=1)[fstix])
 with open(csvfilepath, 'a', newline='') as csvfile:
        res = csv.writer(csvfile)
        res.writerow(['emb','testdegraded','augmentation','trainratio','phclassifier','gendertraincnn','gendertrain','sd','ftlenratio','mfwt','cnwt','spkensacc','sntensacc','spkphmfacc','sntphmfacc','spkcnnacc','sntcnnacc','ensmalspk','phmfcmalspk','cnnmalspk','lblmalspk','ensfemspk','phmfcfemspk','cnnfemspk','lblfemspk','ensmalsent','phmfcmalsent','cnnmalsent','lblmalsent','ensfemsent','phmfcfemsent','cnnfemsent','lblfemsent'])
        res.writerow([emb,testdegraded,augmentation,tpct,phclassifier.__name__,gendertraincnn,gendertrain,sd,ftlenratio,j,l,(msp+fsp)/tlensp,(mst+fst)/tlenst,(amp+afp)/tlensp,(amt+aft)/tlenst,(cmp+cfp)/tlensp,(cmt+cft)/tlenst,np.argmax((j*mphspkoutmfc+l*mcnspkout),axis=1),np.argmax(mphspkoutmfc,axis=1),np.argmax(mcnspkout,axis=1),np.argmax(catlabels,axis=1)[mspix],np.argmax((j*fphspkoutmfc+l*fcnspkout),axis=1),np.argmax(fphspkoutmfc,axis=1),np.argmax(fcnspkout,axis=1),np.argmax(catlabels,axis=1)[fspix],np.argmax((j*mphsntoutmfc+l*mcnsntout),axis=1),np.argmax(mphsntoutmfc,axis=1),np.argmax(mcnsntout,axis=1),np.argmax(catlabels,axis=1)[mstix],np.argmax((j*fphsntoutmfc+l*fcnsntout),axis=1),np.argmax(fphsntoutmfc,axis=1),np.argmax(fcnsntout,axis=1),np.argmax(catlabels,axis=1)[fstix]])
 return

# 
def phoneinput():
    sumdesc1=np.concatenate((np.load(infpath+'summarydescription.npy'),np.load(infpath+'sent_desc.npy').reshape(-1,1)),axis=1)
    catlabels1=ohe().fit_transform(sumdesc1[:,0].reshape(-1, 1)).toarray()   
    return sumdesc1,catlabels1#,np.load(infpath+'mfcc_time_17pad.npy')[:,:,:16].swapaxes(1,2)

def phoneinaug():
    sumdesc1=np.concatenate((np.load(infpath+'summarydescription.npy'),np.load(infpath+'sent_desc.npy').reshape(-1,1)),axis=1)
    catlabels1=ohe().fit_transform(sumdesc1[:,0].reshape(-1, 1)).toarray()
    return np.concatenate((sumdesc1,sumdesc1),axis=0),np.concatenate((catlabels1,catlabels1),axis=0)#,np.concatenate((np.load(infpath+'mfcc_time_17pad.npy')[:,:,:16].swapaxes(1,2),np.load(infpath+'phone4kfiltmfcc.npy').swapaxes(1,2)),axis=0)


def midmfcc():
    feats1=[]
    tmpfeat=np.load(infpath+'mfcc_time_17pad.npy').swapaxes(1,2)
    tmplens=(np.load(infpath+'timemfcclengths.npy')/2).astype(int)
    ix0=0
    for ix in tmplens:
        feats1.append(tmpfeat[ix0][min(ix,16)])
        ix0=ix0+1
    feats1=np.array(feats1)
    return feats1

def fulltimeinput():
    sumdesc1=np.load(infpath+'full_len_sumary_description.npy')
    catlabels1=ohe().fit_transform(sumdesc1[:,0].reshape(-1, 1)).toarray()
    return sumdesc1,catlabels1#,np.load(infpath+'full_len_pdfeatures.npy').swapaxes(1,2)

def fulltimeinaug():
    sumdesc1=np.load(infpath+'full_len_sumary_description.npy')
    catlabels1=ohe().fit_transform(sumdesc1[:,0].reshape(-1, 1)).toarray()
    return np.concatenate((sumdesc1,sumdesc1),axis=0),np.concatenate((catlabels1,catlabels1),axis=0)#,np.concatenate((np.load(infpath+'full_len_pdfeatures.npy').swapaxes(1,2),np.load(infpath+'fulltime4kfiltmfcc.npy').swapaxes(1,2)),axis=0)

def norm():
    train_min = feats.min(axis=(ax[0], ax[1]), keepdims=True) # change to 0,2
    train_max = feats.max(axis=(ax[0], ax[1]), keepdims=True)
    return (feats - train_min)/(train_max - train_min)


# sep21 testing
infpath='../input/phonemedata/'
sumdesc=np.concatenate((np.load(infpath+'summarydescription.npy'),np.load(infpath+'sent_desc.npy').reshape(-1,1)),axis=1)
native=np.unique(sumdesc[:,0])
phonemes=np.unique(sumdesc[:,2])
spkrs=np.unique(sumdesc[:,3],return_index=True)[0]
np.random.shuffle(spkrs)
#feats=np.load(infpath+'mfcc_time_17pad.npy').swapaxes(1,2)
#'''
feats=[]
tmpfeat=np.load(infpath+'mfcc_time_17pad.npy').swapaxes(1,2)
tmplens=(np.load(infpath+'timemfcclengths.npy')/2).astype(int)
ix0=0
for ix in tmplens:
    feats.append(tmpfeat[ix0][min(ix,16)])
    ix0=ix0+1
feats=np.array(feats)
#'''
catlabels=ohe().fit_transform(sumdesc[:,0].reshape(-1, 1)).toarray()   
spklabels=ohe().fit_transform(sumdesc[:,3].reshape(-1, 1)).toarray()

##  dnn

pt=1
inputdim1=feats.shape
outputdim1=catlabels.shape[1]
input_flat1 = Input(shape=(15))
h=input_flat1
output_layer1 = Dense(outputdim1, activation='softmax')(h)
emodel1 = Model(input_flat1,output_layer1)
emodel1.compile(optimizer='Adamax', loss='categorical_crossentropy')
emodel1.save_weights('modeldn10.h5')
emodel1.summary()

input_flat = Input(shape=(20))
h=input_flat
output_layer = Dense(outputdim1, activation='softmax')(h)
dmodel1 = Model(input_flat,output_layer)
dmodel1.compile(optimizer='Adamax', loss='categorical_crossentropy')
dmodel1.save_weights('modeldn20.h5')
dmodel1.summary()

es = EarlyStopping(monitor='loss', mode='min',verbose=1,min_delta=0, patience=pt)
###
#'''

inae = Input(shape=(inputdim1[1:]))
a=inae
###
#a=Conv1D(32, kernel_size=4,strides=1, padding='same')(a)  #
#a=MaxPooling1D()(a)  #
#embae=GlobalAveragePooling1D()(a)  # 
embae=Dense(15,activation='relu')(a)#
oae = Dense(inputdim1[1])(embae)
#oae = Dense(len(spklabels[0]), activation='softmax')(embae)
oap = Dense(1)(embae)
aemdl = Model(inae,[oae,oap])
#aemdl = Model(inae,oae)
iemdl = Model(inae,embae)
aemdl.compile(optimizer='Adamax', loss='mse')
#aemdl.compile(optimizer='Adamax', loss='categorical_crossentropy')
aemdl.save_weights('aemodel.h5')
aemdl.summary()
#'''

aefeats=np.concatenate((np.load(infpath+'gkw.npy'),np.load(infpath+'tmfa.npy')),axis=0)
aefeats.shape


tpct=0.3
tspkrs=spkrs[:int(len(spkrs)*tpct)]
cspkrs=spkrs[int(len(spkrs)*tpct):]
csents=np.unique(sumdesc[:,4][np.in1d(sumdesc[:,3],cspkrs)])

##
phout=np.zeros((len(phonemes),len(cspkrs),len(native)))
phoutsents=np.zeros((len(phonemes),len(csents),len(native)))
phout_emb=np.zeros((len(phonemes),len(cspkrs),len(native)))
phoutsents_emb=np.zeros((len(phonemes),len(csents),len(native)))
phi=0 # best is 'AH' i.e 2 idx
for ph in phonemes[:1]:
    tidx=(np.in1d(sumdesc[:,3],tspkrs))#&(sumdesc[:,2]==ph)
    cidx=(np.in1d(sumdesc[:,3],cspkrs))#&(sumdesc[:,2]==ph)
    print('Training_Joint-gender_Phone',ph)
    acr=np.mean(aefeats,axis=0).reshape(1,-1)#[(sumdesc[:,2]==ph)]
    d=cdist(aefeats,acr, metric='cosine')#[(sumdesc[:,2]==ph)] euclidean
    aemdl.load_weights('aemodel.h5')#[(sumdesc[:,2]==ph)]
    #aemdl.fit(feats,spklabels,epochs=100,batch_size=64,shuffle=True,callbacks=[es])
    aemdl.fit(aefeats,[aefeats,d],epochs=100,batch_size=32,shuffle=True,callbacks=[es])
    embfeats=iemdl.predict(feats)
    emodel1.load_weights('modeldn10.h5')
    hist= emodel1.fit(embfeats[tidx],catlabels[tidx],epochs=100,batch_size=32,shuffle=True,callbacks=[es])
    ennprob = emodel1.predict(embfeats[cidx])
    dmodel1.load_weights('modeldn20.h5')
    hist= dmodel1.fit(feats[tidx],catlabels[tidx],epochs=100,batch_size=32,shuffle=True,callbacks=[es])
    dnnprob = dmodel1.predict(feats[cidx])
    spj=0
    for sp in cspkrs:
        phout[phi][spj]=(np.mean(dnnprob[(sumdesc[:,3][cidx]==sp)],axis=0))
        phout_emb[phi][spj]=(np.mean(ennprob[(sumdesc[:,3][cidx]==sp)],axis=0))
        spj=spj+1
    '''
    spj=0
    for sp in csents:
        phoutsents[phi][spj]=(np.mean(dnnprob[(sumdesc[:,4][cidx]==sp)],axis=0))
        phoutsents_emb[phi][spj]=(np.mean(ennprob[(sumdesc[:,4][cidx]==sp)],axis=0))
        spj=spj+1
    '''
pwts_spkr=[3,3,4,3,1,3,3,3,3,4,4,2,1,1,3]
phout[np.isnan(phout)]=0
phspkout=np.average(phout,axis=0, weights=pwts_spkr)
phout_emb[np.isnan(phout_emb)]=0
phspkout_emb=np.average(phout_emb,axis=0, weights=pwts_spkr)
'''
phoutsents[np.isnan(phoutsents)]=0
phstnout=np.average(phoutsents,axis=0, weights=pwts_spkr)
phoutsents_emb[np.isnan(phoutsents_emb)]=0
phstnout_emb=np.average(phoutsents_emb,axis=0, weights=pwts_spkr)
stix=[sumdesc[:,4].tolist().index(j) for j in csents]
stacc=sum(np.argmax((phstnout),axis=1)==np.argmax(catlabels[stix],axis=1))/len(stix)
stacc_emb=sum(np.argmax((phstnout_emb),axis=1)==np.argmax(catlabels[stix],axis=1))/len(stix)
'''
spix=[sumdesc[:,3].tolist().index(j) for j in cspkrs]
acc=sum(np.argmax((phspkout),axis=1)==np.argmax(catlabels[spix],axis=1))/len(spix)
acc_emb=sum(np.argmax((phspkout_emb),axis=1)==np.argmax(catlabels[spix],axis=1))/len(spix)
phacc=sum(np.argmax((dnnprob),axis=1)==np.argmax(catlabels[cidx],axis=1))/len(catlabels)
phacc_emb=sum(np.argmax((ennprob),axis=1)==np.argmax(catlabels[cidx],axis=1))/len(catlabels)
print(acc,acc_emb,phacc,phacc_emb)
with open('accuracy.csv', 'a', newline='') as csvfile:
    res = csv.writer(csvfile)
    res.writerow([tpct,acc,acc_emb,phacc,phacc_emb])

# %% [code]
#'''

## SIMULATIONS cnn dual accuracy
#
import sys
np.set_printoptions(threshold=sys.maxsize)

infpath='../input/phonemedata/'
csvfilepath='12apr21.csv'
phncsv='phone-acc.csv'


calcphacc='no'
wtdphn='yes' ## change to yes
tfs=22
tms=33
tpct=0.8

l=j=1
gendertraincnn ='joint'
gendertrain ='yes'
ftlenratio=1
augmentation='no' #yes
testdegraded='clean' #clean # both #filter
phclassifier=phcnnprediction
emb='mfcc'

fpwts=fpwts_spkr=mpwts=mpwts_spkr=[3,3,4,3,1,3,3,3,3,4,4,2,1,1,3]

#vpct=0.01 # unused validation set percent for cnn training
#for local iada gpu machine
#import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
#infpath='D:/ali/arrays/'
#csvfilepath='D:/ali/results/finalacc.csv'
#phncsv='D:/ali/results/phone-acc.csv'


for tpct in [0.8,0.4]:#,0.6]:
 for phclassifier in [phcnnprediction]:#['engvowembae_allfeatmean_train1.npy']:# phcnnprediction, ffnprediction]:#   ,svmprediction]:# 
  for gendertrain in ['yes']:#,'joint']:##
   for sd in range(3): # speaker shuffling
    native,phonemes,tmspkrs,tfspkrs,cmspkrs,cfspkrs,cmsents,cfsents,cmsentswav,cfsentswav=spkphnsets()
    gc.collect()
    if 'feats' in globals():
        del sumdesc,catlabels,feats
    gc.collect()  
    if augmentation=='yes':
        sumdesc,catlabels =phoneinaug()
        feats=np.concatenate((np.load(infpath+'mfcc_time_17pad.npy')[:,:,:16].swapaxes(1,2),np.load(infpath+'phone4kfiltmfcc.npy').swapaxes(1,2)),axis=0)
    else:
        sumdesc,catlabels =phoneinput()
        feats=np.load(infpath+'mfcc_time_17pad.npy')[:,:,:16].swapaxes(1,2)
        #feats=np.load(infpath+emb)
    gc.collect()
    mphspkoutmfc,fphspkoutmfc,mphsntoutmfc,fphsntoutmfc=phclassifier()
    #fname='phcnnouts_sd'+str(sd)
    #np.savez_compressed(fname,mphspkoutmfc,fphspkoutmfc,mphsntoutmfc,fphsntoutmfc)  # only when skipping utterence level
    #mcnspkout,fcnspkout,mcnsntout,fcnsntout=mphspkoutmfc,fphspkoutmfc,mphsntoutmfc,fphsntoutmfc  # only when skipping utterence level
    #sumdesc,catlabels =fulltimeinput()  # only when skipping utterence level
    #final_acc_dual() # only when skipping utterence level
#'''
    if 'feats' in globals():
        del sumdesc,catlabels,feats
    gc.collect()
    if augmentation=='yes':
        sumdesc,catlabels =fulltimeinaug()
        feats=np.concatenate((np.load(infpath+'full_len_pdfeatures.npy').swapaxes(1,2),np.load(infpath+'fulltime4kfiltmfcc.npy').swapaxes(1,2)),axis=0)
    else:
        sumdesc,catlabels =fulltimeinput()
        feats=np.load(infpath+'full_len_pdfeatures.npy').swapaxes(1,2)
    gc.collect()
    for ftlenratio in [1]:#,2]: # feature time ratio for long term cnn
     for gendertraincnn in ['joint']:#,'yes']:##
      mcnspkout,fcnspkout,mcnsntout,fcnsntout=ltcnnprediction()
      #fname='ltcnnouts_sd'+str(sd)
      #np.savez_compressed(fname,mcnspkout,fcnspkout,mcnsntout,fcnsntout)
      gc.collect()
      for j in [2]:#[0.25,0.5,0.75,1,2,3,4]:#weight for ST
        l=1 # weight for LT
        final_acc_dual()

'''
#feats=np.argsort(-(np.load(infpath+'midphnmelspec128.npy')),axis=1)[:,:5]
#feats=np.argsort(-(np.load(infpath+'midphnmelspec64.npy')),axis=1)[:,:5]
#feats=np.argsort(-(np.load(infpath+'midphnmelspec64.npy')),axis=1)[:,:10]################
#'''##########

# %% [code]
# accdist Measure
def accdist():
 #'''
 # acc dist features
 #clf = svm.SVC(probability=True)
 clf = svm.SVC()
 from scipy.spatial.distance import pdist
 trainfeataccdist=[]
 trainlblaccdist=[]
 checkfeataccdist=[]
 checklblaccdist=[]
 mcheckfeataccdist=[]
 mchecklblaccdist=[]
 fcheckfeataccdist=[]
 fchecklblaccdist=[]
 tspkrs=np.concatenate((tmspkrs,tfspkrs),axis=0)
 for sp in tspkrs:
     phmean=[]
     for ph in phonemes:
         phmean.append(np.mean(feats[(sumdesc[:,3]==sp)&(sumdesc[:,2]==ph)],axis=0))
     trainfeataccdist.append(pdist(phmean))
     trainlblaccdist.append(catlabels[(sumdesc[:,3]==sp)&(sumdesc[:,2]==ph)][0])
 '''
 cspkrs=np.concatenate((cmspkrs,cfspkrs),axis=0)
 for sp in cspkrs:
     phmean=[]
     for ph in phonemes:
         phmean.append(np.mean(feats[(sumdesc[:,3]==sp)&(sumdesc[:,2]==ph)],axis=0))
     checkfeataccdist.append(pdist(phmean))
     checklblaccdist.append(catlabels[(sumdesc[:,3]==sp)&(sumdesc[:,2]==ph)][0])
 '''
 for sp in cmspkrs:
     phmean=[]
     for ph in phonemes:
         phmean.append(np.mean(feats[(sumdesc[:,3]==sp)&(sumdesc[:,2]==ph)],axis=0))
     mcheckfeataccdist.append(pdist(phmean))
     mchecklblaccdist.append(catlabels[(sumdesc[:,3]==sp)&(sumdesc[:,2]==ph)][0])
 
 for sp in cfspkrs:
     phmean=[]
     for ph in phonemes:
         phmean.append(np.mean(feats[(sumdesc[:,3]==sp)&(sumdesc[:,2]==ph)],axis=0))
     fcheckfeataccdist.append(pdist(phmean))
     fchecklblaccdist.append(catlabels[(sumdesc[:,3]==sp)&(sumdesc[:,2]==ph)][0])
 trainfeataccdist=np.array(trainfeataccdist)
 trainfeataccdist[np.isnan(trainfeataccdist)]=0
 mcheckfeataccdist=np.array(mcheckfeataccdist)
 mcheckfeataccdist[np.isnan(mcheckfeataccdist)]=0
 fcheckfeataccdist=np.array(fcheckfeataccdist)
 fcheckfeataccdist[np.isnan(fcheckfeataccdist)]=0
 #print(tspkrs.shape,cmspkrs.shape)
 #print(mcheckfeataccdist.shape,fcheckfeataccdist.shape)
 print('SVM_Training-accdist')
 clf.fit(trainfeataccdist, np.argmax(trainlblaccdist,axis=1))
 #msvmprob = clf.predict_proba(mcheckfeataccdist)
 #fsvmprob = clf.predict_proba(fcheckfeataccdist)
 msvmprob = clf.predict(mcheckfeataccdist)
 fsvmprob = clf.predict(fcheckfeataccdist)
 return msvmprob,fsvmprob

# '''


# %% [code]
'''
# accdist accuracy calculation
infpath='../input/phonemedata/'
csvfilepath='phcnn16time_wtd.csv'
tfs=22
tms=33
tpct=0.8

#for tpct in [0.8]:#0.7,0.9]:
for sd in range(5,10): # speaker shuffling
    native,phonemes,tmspkrs,tfspkrs,cmspkrs,cfspkrs,cmsents,cfsents,cmsentswav,cfsentswav=spkphnsets()
    gc.collect()
    sumdesc=np.concatenate((np.load(infpath+'summarydescription.npy'),np.load(infpath+'sent_desc.npy').reshape(-1,1)),axis=1)
    catlabels=ohe().fit_transform(sumdesc[:,0].reshape(-1, 1)).toarray()
    feats=midmfcc()
    gc.collect()
    mphspkoutmfc,fphspkoutmfc=accdist()
    mspix=[sumdesc[:,3].tolist().index(j) for j in cmspkrs]
    fspix=[sumdesc[:,3].tolist().index(j) for j in cfspkrs]
    tlensp=len(mspix)+len(fspix)
    msp=sum(mphspkoutmfc==np.argmax(catlabels,axis=1)[mspix])
    fsp=sum(fphspkoutmfc==np.argmax(catlabels,axis=1)[fspix])
    print((msp+fsp)/tlensp,sd)
    
'''

# %%