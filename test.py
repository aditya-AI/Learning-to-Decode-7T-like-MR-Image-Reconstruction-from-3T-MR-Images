import os
import cv2
from keras.layers import Input,Dense,Flatten,Dropout,merge,Reshape,Conv2D,MaxPooling2D,UpSampling2D,Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.models import Model,Sequential
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adadelta, RMSprop,SGD,Adam
from keras import regularizers
from keras import backend as K
import numpy as np
import scipy.misc
import numpy.random as rng
from PIL import Image, ImageDraw, ImageFont
from sklearn.utils import shuffle
import nibabel as nib
from sklearn.cross_validation import train_test_split
import math
aaa=np.loadtxt('../maxANDmin.txt')

x,y = 173,173
z=range(78,129)
sliced_z = 51
full_z = 207
new_z = len(z)
resizeTo=176
batch_size = 32
inChannel = outChannel = 1
epoch = 2000
input_shape=(x,y,inChannel)
input_img = Input(shape = (resizeTo, resizeTo, inChannel))	

train_matrix = []
test_matrix = []
inp = "../ground3T/"
out = "../ground7T/"
ff = os.listdir("../test_crossval1")
save = "../Result_nii_crossval1/"
folder_ground = os.listdir("../test_g_crossval1")	
t = []
ToPredict_images=[]
Tgx_images=[]
predict_matrix=[]
Tgy_images=[]

ground_images=[]
Tgxx_images_g=[]
Tgyx_images_g=[]
Tgxx_images=[]
Tgyx_images=[]
Tgxx_images=np.asarray(Tgxx_images)
Tgyx_images=np.asarray(Tgyx_images)
Tgx_images_g=[]
ground_matrix=[]
Tgy_images_g=[]
# UPLOAD THE TEST 3t AND TEST 7t IMAGES (GROUND)
for f in ff:
	temp = np.zeros([resizeTo,full_z,resizeTo])
	a = nib.load("../test_crossval1/" + f)
	affine = a.affine
	a = a.get_data()
	temp[3:,:,3:] = a
	a = temp
	Tgxx_images=[]
	Tgyx_images=[]
	#for k in range(full_z):	
		#Tgxx_images.append(cv2.Sobel(np.reshape(a[:,k,:],[resizeTo,resizeTo]),cv2.CV_64F,1,0,ksize=5))
		#Tgyx_images.append(cv2.Sobel(np.reshape(a[:,k,:],[resizeTo,resizeTo]),cv2.CV_64F,0,1,ksize=5))
	#Tgxx_images=np.asarray(Tgxx_images)
	#Tgyx_images=np.asarray(Tgyx_images)
	for j in range(full_z):
		predict_matrix.append(a[:,j,:])
		#Tgx_images.append(Tgxx_images[j,:,:])
		#Tgy_images.append(Tgyx_images[j,:,:])

predict_matrix = np.asarray(predict_matrix)
Tgx_images=np.asarray(Tgx_images)
Tgy_images=np.asarray(Tgy_images)
ToPredict_images=np.zeros(shape=[(predict_matrix.shape[0]),(predict_matrix.shape[1]),(predict_matrix.shape[2]),(1)])

for i in range(predict_matrix.shape[0]):
	ToPredict_images[i,:,:,0] = predict_matrix[i,:,:].reshape(resizeTo,resizeTo)
	#ToPredict_images[i,:,:,1] = Tgx_images[i,:,:].reshape(resizeTo,resizeTo)
	#ToPredict_images[i,:,:,2] = Tgy_images[i,:,:].reshape(resizeTo,resizeTo)


for f in ff:
	temp = np.zeros([resizeTo,full_z,resizeTo])
	a = nib.load("../test_g_crossval1/" + f)
	affine = a.affine
	a = a.get_data()
	temp[3:,:,3:] = a
	a = temp
	Tgxx_images_g=[]
	Tgyx_images_g=[]
	#for k in range(full_z):	
		#Tgxx_images_g.append(cv2.Sobel(np.reshape(a[:,k,:],[resizeTo,resizeTo]),cv2.CV_64F,1,0,ksize=5))
		#Tgyx_images_g.append(cv2.Sobel(np.reshape(a[:,k,:],[resizeTo,resizeTo]),cv2.CV_64F,0,1,ksize=5))
	#Tgxx_images_g=np.asarray(Tgxx_images_g)
	#Tgyx_images_g=np.asarray(Tgyx_images_g)
	for j in range(full_z):
		ground_matrix.append(a[:,j,:])
		#Tgx_images_g.append(Tgxx_images_g[j,:,:])
		#Tgy_images_g.append(Tgyx_images_g[j,:,:])

ground_matrix=np.asarray(ground_matrix)
Tgx_images_g=np.asarray(Tgx_images_g)
Tgy_images_g=np.asarray(Tgy_images_g)
ground_images=np.zeros(shape=[(ground_matrix.shape[0]),(ground_matrix.shape[1]),(ground_matrix.shape[2]),(1)])


for i in range(ground_matrix.shape[0]):
	ground_images[i,:,:,0] = ground_matrix[i,:,:].reshape(resizeTo,resizeTo)
	#ground_images[i,:,:,1] = Tgx_images_g[i,:,:].reshape(resizeTo,resizeTo)
	#ground_images[i,:,:,2] = Tgy_images_g[i,:,:].reshape(resizeTo,resizeTo)


ground_images = np.asarray(ground_images)
ground_images = ground_images.astype('float32')
mx = aaa[0]#np.max(ground_images)
mn = aaa[1]#np.min(ground_images)
ground_images[:,:,:,0] = (ground_images[:,:,:,0] - mn ) / (mx - mn)
#ground_images[:,:,:,1] = (ground_images[:,:,:,1] - aaa[3] ) / (aaa[2] - aaa[3])
#ground_images[:,:,:,2] = (ground_images[:,:,:,2] - aaa[5] ) / (aaa[4] - aaa[5])


ToPredict_images = np.asarray(ToPredict_images)
ToPredict_images = ToPredict_images.astype('float32')
mx = aaa[0]#np.max(ToPredict_images)
mn = aaa[1]#np.min(ToPredict_images)
ToPredict_images[:,:,:,0] = (ToPredict_images[:,:,:,0] - mn ) / (mx - mn)
#ToPredict_images[:,:,:,1] = (ToPredict_images[:,:,:,1] - aaa[3] ) / (aaa[2] - aaa[3])
#ToPredict_images[:,:,:,2] = (ToPredict_images[:,:,:,2] - aaa[5] ) / (aaa[4] - aaa[5])


x,y = 173,173
z=range(78,129)
new_z = len(z)
full_z=207
resizeTo=176
batch_size = 64
inChannel = outChannel = 1
input_shape=(x,y,inChannel)
input_img = Input(shape = (resizeTo, resizeTo, inChannel))	
inp = "../ground3T/"
out = "../ground7T/"     
train_matrix = []
test_matrix = []
Lgx_images = []
Hgx_images = []
Lgy_images = []
Hgy_images = []

folder1 = os.listdir(inp)

for f in folder1:
	Lgxx_images = []
	Hgxx_images = []
	Lgyx_images = []
	Hgyx_images = []
	temp = np.zeros([resizeTo,full_z,resizeTo])
	a = nib.load(inp + f)
	a = a.get_data()
	temp[3:,:,3:] = a
	a = temp
	b = nib.load(out + f)
	b = b.get_data()
	temp[3:,:,3:] = b
	b = temp
	for k in range(full_z):	
		Lgxx_images.append(cv2.Sobel(np.reshape(a[:,k,:],[resizeTo,resizeTo]),cv2.CV_64F,1,0,ksize=5))
		Hgxx_images.append(cv2.Sobel(np.reshape(b[:,k,:],[resizeTo,resizeTo]),cv2.CV_64F,1,0,ksize=5))
		Lgyx_images.append(cv2.Sobel(np.reshape(a[:,k,:],[resizeTo,resizeTo]),cv2.CV_64F,0,1,ksize=5))
		Hgyx_images.append(cv2.Sobel(np.reshape(b[:,k,:],[resizeTo,resizeTo]),cv2.CV_64F,0,1,ksize=5))
	Lgxx_images=np.asarray(Lgxx_images)
	Hgxx_images=np.asarray(Hgxx_images)
	Lgyx_images=np.asarray(Lgyx_images)
	Hgyx_images=np.asarray(Hgyx_images)
	for j in range(full_z):
		train_matrix.append(a[:,j,:])
		test_matrix.append(b[:,j,:])
		Lgx_images.append(Lgxx_images[j,:,:])
		Hgx_images.append(Hgxx_images[j,:,:])
		Lgy_images.append(Lgyx_images[j,:,:])
		Hgy_images.append(Hgyx_images[j,:,:])


train_matrix = np.asarray(train_matrix)
train_matrix = train_matrix.astype('float32')
m = np.max(train_matrix)
mi = np.min(train_matrix)
train_matrix = (train_matrix - mi) / (m - mi)

test_matrix = np.asarray(test_matrix)
test_matrix = test_matrix.astype('float32')
test_matrix = (test_matrix - mi) / (m - mi)

Lgx_images = np.asarray(Lgx_images)
Lgx_images = Lgx_images.astype('float32')
m = np.max(Lgx_images)
mi = np.min(Lgx_images)
Lgx_images = (Lgx_images - mi) / (m - mi)

Hgx_images = np.asarray(Hgx_images)
Hgx_images = Hgx_images.astype('float32')
Hgx_images = (Hgx_images - mi) / (m - mi)

Lgy_images = np.asarray(Lgy_images)
Lgy_images = Lgy_images.astype('float32')
m = np.max(Lgy_images)
mi = np.min(Lgy_images)
Lgy_images = (Lgy_images - mi) / (m - mi)

Hgy_images = np.asarray(Hgy_images)
Hgy_images = Hgy_images.astype('float32')
Hgy_images = (Hgy_images - mi) / (m - mi)

augmented_images=np.zeros(shape=[(train_matrix.shape[0]),(train_matrix.shape[1]),(train_matrix.shape[2]),(1)])
Haugmented_images=np.zeros(shape=[(train_matrix.shape[0]),(train_matrix.shape[1]),(train_matrix.shape[2]),(1)])

for i in range(train_matrix.shape[0]):
	augmented_images[i,:,:,0] = train_matrix[i,:,:].reshape(resizeTo,resizeTo)
	#augmented_images[i,:,:,1] = Lgx_images[i,:,:].reshape(resizeTo,resizeTo)
	#augmented_images[i,:,:,2] = Lgy_images[i,:,:].reshape(resizeTo,resizeTo)
	Haugmented_images[i,:,:,0] = test_matrix[i,:,:].reshape(resizeTo,resizeTo)
	#Haugmented_images[i,:,:,1] = Hgx_images[i,:,:].reshape(resizeTo,resizeTo)
	#Haugmented_images[i,:,:,2] = Hgy_images[i,:,:].reshape(resizeTo,resizeTo)


data,Label = shuffle(augmented_images,Haugmented_images, random_state=2)
X_train, X_test, y_train, y_test = train_test_split(data, Label, test_size=0.125, random_state=2)
X_test = np.array(X_test)
#X_test = np.expand_dims(X_test,3)
y_test = np.array(y_test)
#y_test = np.expand_dims(y_test,3)
X_test = X_test.astype('float32')
y_test = y_test.astype('float32')
X_train = np.array(X_train)
#m = np.max(X_train)
#mi = np.min(X_train)
#X_train = np.expand_dims(X_train,3)
y_train = np.array(y_train)
y_train = np.expand_dims(y_train,3)
X_train = X_train.astype('float32')
y_train = y_train.astype('float32')

#X_train = (X_train - mi) / (m - mi)
#X_test = (X_test - mi) / (m - mi)
#y_test = (y_test - mi) / (m - mi)
#y_train = (y_train - mi) / (m - mi)



def autoencoder(input_img):
	conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
	conv1 = BatchNormalization()(conv1)
	conv1 = Conv2D(32, (3,3), activation='relu', padding='same')(conv1)
	conv1 = BatchNormalization()(conv1)
	conv1 = Conv2D(32, (3,3), activation='relu', padding='same')(conv1)
	conv1 = BatchNormalization()(conv1)
	pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
	conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
	conv2 = BatchNormalization()(conv2)
	conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
	conv2 = BatchNormalization()(conv2)
	conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
	conv2 = BatchNormalization()(conv2)
	pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
	conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
	conv3 = BatchNormalization()(conv3)
	conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
	conv3 = BatchNormalization()(conv3)
	conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
	conv3 = BatchNormalization()(conv3)
	pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
	conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
	conv4 = BatchNormalization()(conv4)
	conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
	conv4 = BatchNormalization()(conv4)
	conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
	conv4 = BatchNormalization()(conv4)
	#pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
	conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv4)
	conv5 = BatchNormalization()(conv5)
	conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
	conv5 = BatchNormalization()(conv5)
	conv5 = Conv2D(512, (3, 3), activation='sigmoid', padding='same')(conv5)
	conv5 = BatchNormalization()(conv5)




	#up6 = UpSampling2D((2,2))(conv5)
	up6 = merge([conv5, conv4], mode='concat', concat_axis=3)
	conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
	conv6 = BatchNormalization()(conv6)
	conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)
	conv6 = BatchNormalization()(conv6)
	conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)
	conv6 = BatchNormalization()(conv6)
	up7 = UpSampling2D((2,2))(conv6)
	up7 = merge([up7, conv3], mode='concat', concat_axis=3)
	conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
	conv7 = BatchNormalization()(conv7)
	conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)
	conv7 = BatchNormalization()(conv7)
	conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)
	conv7 = BatchNormalization()(conv7)
	up8 = UpSampling2D((2,2))(conv7)
	up8 = merge([up8, conv2], mode='concat', concat_axis=3)
	conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
	conv8 = BatchNormalization()(conv8)
	conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)
	conv8 = BatchNormalization()(conv8)
	conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)
	conv8 = BatchNormalization()(conv8)
	up9 = UpSampling2D((2,2))(conv8)
	up9 = merge([up9, conv1], mode='concat', concat_axis=3)
	conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
	conv9 = BatchNormalization()(conv9)
	conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)
	conv9 = BatchNormalization()(conv9)	
	conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)
	conv9 = BatchNormalization()(conv9)
	decoded_1 = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(conv9)
	return decoded_1


def root_mean_sq_GxGy(y_t, y_p):
	a1=1
	zero = tf.constant(0, dtype=tf.float32)
	where = tf.not_equal(y_t, zero)
	a_t=tf.boolean_mask(y_t,where,name='boolean_mask')
	a_p=tf.boolean_mask(y_p,where,name='boolean_mask')
#	print tf.shape(a_t)	
#	indices = tf.where(where)
#	count=tf.count_nonzero(y_t)
#	temp=K.sum(K.square(y_t-y_p))
#	count=tf.cast(count,tf.float32)
#	return a1*(K.sqrt(tf.divide(temp,count)))#+(a2*(K.sqrt(K.mean(K.square((PIx-Ix))))))+(a3*(K.sqrt(K.mean(K.square((PIy-Iy))))))
	return a1*(K.sqrt(K.mean((K.square(a_t-a_p)))))#+(a2*(K.sqrt(K.mean(K.square((PIx-Ix))))))+(a3*(K.sqrt(K.mean(K.square((PIy-Iy))))))'''

autoencoder_1 = Model(input_img, autoencoder(input_img))
#autoencoder.compile(loss=root_mean_sq_GxGy, optimizer = SGD())

autoencoder_1.load_weights("../Model/CROSSVAL1/OLD_CROSSVAL1_AE1.h5")			  

autoencoder_2 = Model(input_img, autoencoder(input_img))
#autoencoder.compile(loss=root_mean_sq_GxGy, optimizer = SGD())

autoencoder_2.load_weights("../Model/CROSSVAL1/OLD_CROSSVAL1_AE2.h5")

autoencoder_3 = Model(input_img, autoencoder(input_img))
#autoencoder.compile(loss=root_mean_sq_GxGy, optimizer = SGD())

autoencoder_3.load_weights("../Model/CROSSVAL1/OLD_CROSSVAL1_AE3.h5")

#autoencoder_3 = Model(input_img, autoencoder(input_img))
#autoencoder.compile(loss=root_mean_sq_GxGy, optimizer = SGD())

#autoencoder_3.load_weights("../Model/BEST_simple_ae_complete_slices_MICCAI_413.h5")

#autoencoder.load_weights("../Model/simple_ae_complete_slices_MICCAI.h5")
'''
convout1_f = K.function([autoencoder_2.layers[0].input,K.learning_phase()], [autoencoder_2.layers[51].output])
print (autoencoder_1.layers[5])
i = 369

# Visualize the first layer of convolutions on an input image
X = X_train[i:i+1,:,:,:]

C1 = convout1_f([X,1])[0]
C1 = np.squeeze(C1)
print("C1 shape : ", C1.shape)
#C1 = C1.reshape(1,88,88,64)
print(type(C1))
#visualize(C1)
from PIL import Image
#C1 = C1.reshape(-1,88,88)
#print (C1.shape[2])

for i in range(C1.shape[2]):
	#C1[:,:,i] = C1[:,:,i].resize((200, 200), Image.ANTIALIAS) 
	result = Image.fromarray((C1[:,:,i] * 255).astype(np.uint8))
	result = result.resize((200, 200), Image.ANTIALIAS) 
	result.save('../Activation_Maps/AE1/CONV6/' + str(i) + '.jpg')
'''
mse= np.zeros([12,3,3])
psnr= np.zeros([12,3,3])
i=0

'''
z,x,y=np.where(ToPredict_images[:,:,:,0]<0.1)	
ToPredict_images[z,x,y,0]=0
z,x,y=np.where(ground_images[:,:,:,0]<0.1)	
ground_images[z,x,y,0]=0
z,x,y=np.where(ToPredict_images[:,:,:,1]<0.001)	
ToPredict_images[z,x,y,1]=0
z,x,y=np.where(ground_images[:,:,:,1]<0.001)	
ground_images[z,x,y,1]=0
z,x,y=np.where(ToPredict_images[:,:,:,2]<0.001)	
ToPredict_images[z,x,y,2]=0
z,x,y=np.where(ground_images[:,:,:,2]<0.001)	
ground_images[z,x,y,2]=0
'''


#toPredict_images = np.negative(np.log10(ToPredict_images))
#Ground_images = np.negative(np.log10(ground_images))

'''for jj in range(561):
	temp=np.zeros([3*resizeTo,resizeTo])
	temp[:resizeTo,:] = ToPredict_images[jj,:,:,0]
	temp[resizeTo:resizeTo*2,:] = ground_images[jj,:,:,0]
	scipy.misc.imsave('results_test/' + str(jj) + 'ground.jpg', 255*ground_images[jj,:,:,0])
	scipy.misc.imsave('results_test/' + str(jj) + 'Predict.jpg', 255*ToPredict_images[jj,:,:,0])
	a=np.asarray(ToPredict_images[jj,:,:,:])
	a=np.expand_dims(a,0)
	decoded_imgs = autoencoder.predict(a)
	decoded_imgs = np.squeeze(decoded_imgs,0)
	temp[resizeTo*2:resizeTo*3,:] = decoded_imgs[:,:,0]
	scipy.misc.imsave('results_test/' + str(jj) + 'decoded.jpg', 255*decoded_imgs[:,:,0])
	scipy.misc.imsave('results_test/' + str(jj) + '3T7T.jpg', 255*temp)'''
	

j=0
for j in range(11):
	decoded_imgs_1 = autoencoder_1.predict(ToPredict_images[i:i+207,:,:,:])
	decoded_imgs_2 = autoencoder_2.predict(ToPredict_images[i:i+207,:,:,:])
	decoded_imgs_3 = autoencoder_3.predict(ToPredict_images[i:i+207,:,:,:])
	decoded_imgs = np.mean( np.array([ decoded_imgs_1, decoded_imgs_2,decoded_imgs_3 ]), axis=0 )
	for channel in range(1):
		mse[j,0,channel]=  np.mean((ground_images[i:i+207,:,:,channel] - decoded_imgs[:,:,:,channel]) ** 2)
		check = math.sqrt(mse[j,0,channel])
		#print "{}" ".Inference".format(j+1)
		#print ('\n')	
		#print "GROUND VS DECODED"
		#print (check)
		psnr[j,0,channel] = 20 * math.log10( 1.0 / math.sqrt(mse[j,0,channel]))
		#print (psnr[j,0,channel])
		#print ("\n")
		mse[j,1,channel]=  np.mean((ground_images[i:i+207,:,:,channel] - ToPredict_images[i:i+207,:,:,channel]) ** 2)
		checklh = math.sqrt(mse[j,1,channel])
		#print "GROUND VS INPUT LR"
		#print (checklh)
		psnr[j,1,channel] = 20 * math.log10( 1.0 / math.sqrt(mse[j,1,channel]))
		#print (psnr[j,1,channel])
		#print ("\n")
		mse[j,2,channel] =  np.mean((ToPredict_images[i:i+207,:,:,channel] - decoded_imgs[:,:,:,channel]) ** 2)
		checklt = math.sqrt(mse[j,2,channel])
		#print "INPUT LR VS DECODED"
		#print (checklt)
		psnr[j,2,channel] = 20 * math.log10( 1.0 / math.sqrt(mse[j,2,channel]))
		#print (psnr[j,2,channel])
		#print ("--------------------------------------------------------")
		#print ("\n")
	#print "SEE WHAT HAPPENS NEXT"
	#obj = nib.Nifti1Image(decoded_imgs, affine)
	#string =str(j)+'test_3decoders.nii'
	#nib.save(obj, save + string)
	obj = nib.Nifti1Image(decoded_imgs, affine)
	string =str(j)+'_crossval1.nii'
	nib.save(obj, save + string)
#	ground_images1[i:i+51,:,:,0]=(ground_images[i:i+51,:,:,0]*(aaa[0]-aaa[1]))+aaa[1]
#	ground_images1[i:i+51,:,:,1]=(ground_images[i:i+51,:,:,1]*(aaa[2]-aaa[3]))+aaa[3]
#	ground_images1[i:i+51,:,:,2]=(ground_images[i:i+51,:,:,2]*(aaa[4]-aaa[5]))+aaa[5]
	obj = nib.Nifti1Image(ground_images[i:i+207,:,:,:], affine)
	string =str(j)+'_ground_images_crossval1.nii'
	nib.save(obj, save + string)
#	ToPredict_images1[i:i+51,:,:,0]=(ToPredict_images1[i:i+51,:,:,0]*(aaa[0]-aaa[1]))+aaa[1]
#	ToPredict_images1[i:i+51,:,:,1]=(ToPredict_images1[i:i+51,:,:,1]*(aaa[2]-aaa[3]))+aaa[3]
#	ToPredict_images1[i:i+51,:,:,2]=(ToPredict_images1[i:i+51,:,:,2]*(aaa[4]-aaa[5]))+aaa[5]
	obj = nib.Nifti1Image(ToPredict_images[i:i+207,:,:,:], affine)
	string =str(j)+'_ToPredict_images_crossval1.nii'
	nib.save(obj, save + string)
	i = i+207

'''
	temp = np.zeros([resizeTo*3,resizeTo*3])
	temp[:resizeTo,:resizeTo] = ToPredict_images[i+78,:,:,0]
	temp[resizeTo:resizeTo*2,:resizeTo] = ToPredict_images[i+78,:,:,1]
	temp[resizeTo*2:,:resizeTo] = ToPredict_images[i+78,:,:,2]
	temp[:resizeTo,resizeTo:resizeTo*2] = ground_images[i+78,:,:,0]
	temp[resizeTo:resizeTo*2,resizeTo:resizeTo*2] = ground_images[i+78,:,:,1]
	temp[resizeTo*2:,resizeTo:resizeTo*2] = ground_images[i+78,:,:,2]
	temp[:resizeTo,2*resizeTo:] = decoded_imgs[78,:,:,0]
	temp[resizeTo:resizeTo*2,2*resizeTo:] = decoded_imgs[78,:,:,1]
	temp[resizeTo*2:,2*resizeTo:] = decoded_imgs[78,:,:,2]
	temp = temp*255
	
	#scipy.misc.imsave('../AllResults/results_test_all_slices/' + str(j) + '.jpg', temp)
	i=i+207
'''
'''
print "Channel 0:"
print "Ground Vs Decoded:    	"+ "Average PSNR: "+"     	"+str(np.mean(psnr[:,0,0])) +"      "	+"Average MSE:"+	str(np.mean(mse[:,0,0]))
print "Ground VS LR: 	 	"+ "Average PSNR: "+"     	"+str(np.mean(psnr[:,1,0])) +"      "	+"Average MSE:"+	str(np.mean(mse[:,1,0]))
print "Decoded VS LR:      	"+ "Average PSNR: "+"           "+str(np.mean(psnr[:,2,0])) +"      "	+"Average MSE:"+	str(np.mean(mse[:,2,0]))
print "\n"

print "Channel 1:"
print "Ground Vs Decoded:  	"+ "Average PSNR: "+"         " +str(np.mean(psnr[:,0,1]))+ "      "	+"Average MSE:"+	str(np.mean(mse[:,0,1]))
print "Ground VS LR:  	   	"+ "Average PSNR: "+"         " +str(np.mean(psnr[:,1,1]))+ "      "	+"Average MSE:"+	str(np.mean(mse[:,1,1]))
print "Decoded VS LR:      	"+ "Average PSNR: "+"         " +str(np.mean(psnr[:,2,1]))+ "      "	+"Average MSE:"+	str(np.mean(mse[:,2,1]))
print "\n"

print "Channel 2:"
print "Ground Vs Decoded: 	"+ "Average PSNR: "+"         " +str(np.mean(psnr[:,0,2]))+ "      " 	+"Average MSE:"+	str(np.mean(mse[:,0,2]))
print "Ground VS LR: 	 	"+ "Average PSNR: "+"         " +str(np.mean(psnr[:,1,2]))+ "      "	+"Average MSE:"+	str(np.mean(mse[:,1,2]))
print "Decoded VS LR: 		"+ "Average PSNR: "+"         " +str(np.mean(psnr[:,2,2]))+ "      "	+"Average MSE:"+	str(np.mean(mse[:,2,2]))
'''

np.savetxt('psnr_all_slices.txt',psnr[:,:,0])

