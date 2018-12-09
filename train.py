import numpy 
import tensorflow	
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
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
import tensorflow as tf 
import matplotlib.pyplot as plt
import numpy as np
import math
from keras.models import model_from_json

x,y = 173,173
z=range(78,129)
sliced_z = 51
full_z = 207
new_z = len(z)
resizeTo=176
batch_size = 32
inChannel = outChannel = 1
input_shape=(x,y,inChannel)
input_img = Input(shape = (resizeTo, resizeTo, inChannel))	
inp = "../CROSS_VAL_DATA/ground3T/"
out = "../CROSS_VAL_DATA/ground7T/"     
train_matrix = []
test_matrix = []
Lgx_images = []
Hgx_images = []
Lgy_images = []
Hgy_images = []
min_max = np.loadtxt('../maxANDmin.txt')

folder1 = os.listdir(inp)
#folder2 = os.listdir("../ground3T_Masks")

for f in folder1:
	Lgxx_images = []
	Lgyx_images = []
	temp = np.zeros([resizeTo,full_z,resizeTo])
	a = nib.load(inp + f)
	a = a.get_data()
	temp[3:,:,3:] = a
	a = temp
	#for k in range(full_z):	
		#Lgxx_images.append(cv2.Sobel(np.reshape(a[:,k,:],[resizeTo,resizeTo]),cv2.CV_64F,1,0,ksize=5))
		#Lgyx_images.append(cv2.Sobel(np.reshape(a[:,k,:],[resizeTo,resizeTo]),cv2.CV_64F,0,1,ksize=5))
	#Lgxx_images=np.asarray(Lgxx_images)
	#Lgyx_images=np.asarray(Lgyx_images)
	for j in range(full_z):
		train_matrix.append(a[:,j,:])
		#Lgx_images.append(Lgxx_images[j,:,:])
		#Lgy_images.append(Lgyx_images[j,:,:])


for f in folder1:
	Hgxx_images = []
	Hgyx_images=[]
	temp = np.zeros([resizeTo,full_z,resizeTo])
	b = nib.load(out + f)
	b = b.get_data()
	temp[3:,:,3:] = b
	b = temp
	#for k in range(full_z):	
		#Hgxx_images.append(cv2.Sobel(np.reshape(b[:,k,:],[resizeTo,resizeTo]),cv2.CV_64F,1,0,ksize=5))
		#Hgyx_images.append(cv2.Sobel(np.reshape(b[:,k,:],[resizeTo,resizeTo]),cv2.CV_64F,0,1,ksize=5))
	#Hgxx_images=np.asarray(Hgxx_images)
	#Hgyx_images=np.asarray(Hgyx_images)
	for j in range(full_z):
		test_matrix.append(b[:,j,:])
		#Hgx_images.append(Hgxx_images[j,:,:])
		#Hgy_images.append(Hgyx_images[j,:,:])

train_matrix = np.asarray(train_matrix)
train_matrix = train_matrix.astype('float32')
m = min_max[0]
mi = min_max[1]
train_matrix = (train_matrix - mi) / (m - mi)

test_matrix = np.asarray(test_matrix)
test_matrix = test_matrix.astype('float32')
test_matrix = (test_matrix - mi) / (m - mi)

Lgx_images = np.asarray(Lgx_images)
Lgx_images = Lgx_images.astype('float32')
m = min_max[2]
mi = min_max[3]
Lgx_images = (Lgx_images - mi) / (m - mi)

Hgx_images = np.asarray(Hgx_images)
Hgx_images = Hgx_images.astype('float32')
Hgx_images = (Hgx_images - mi) / (m - mi)

Lgy_images = np.asarray(Lgy_images)
Lgy_images = Lgy_images.astype('float32')
m = min_max[4]
mi = min_max[5]
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
'''
temp=np.zeros([176,176*2])
temp[:176,:176]=augmented_images[51,:,:,0]
temp[:176,176:176*2]=Haugmented_images[51,:,:,0]
scipy.misc.imsave('../AllResults/1_results_slices51_Masked_MSE/' + 'SeeDiffIn3T&And7T' + '.jpg', temp)
'''

data,Label = shuffle(augmented_images,Haugmented_images, random_state=2)
X_train, X_test, y_train, y_test = train_test_split(data, Label, test_size=0.2, random_state=2)
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
#y_train = np.expand_dims(y_train,3)
X_train = X_train.astype('float32')
y_train = y_train.astype('float32')

print (X_train.shape)


def encoder_1(input_img):
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
	return conv5,conv4,conv3,conv2,conv1

def decoder_1(conv5,conv4,conv3,conv2,conv1):
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
	
	a_t=tf.boolean_mask(y_t,where,name='boolean_mask')
	a_p=tf.boolean_mask(y_p,where,name='boolean_mask')
	return a1*(K.sqrt(K.mean((K.square(a_t-a_p)))))


conv5,conv4,conv3,conv2,conv1 = encoder_1(input_img)

autoencoder_1 = Model(input_img, decoder_1(conv5,conv4,conv3,conv2,conv1))
autoencoder_1.compile(loss=root_mean_sq_GxGy, optimizer = RMSprop())

autoencoder_2 = Model(input_img, decoder_2(conv5,conv4,conv3,conv2,conv1))
autoencoder_2.compile(loss=root_mean_sq_GxGy, optimizer = RMSprop())

autoencoder_3 = Model(input_img, decoder_3(conv5,conv4,conv3,conv2,conv1))
autoencoder_3.compile(loss=root_mean_sq_GxGy, optimizer = RMSprop())

#autoencoder_1.summary()


autoencoder_1.load_weights("../Model/OLD_1_encoder_3_decoders_complete_slices_single_channel_second_time/AE1_BEST.h5")
autoencoder_2.load_weights("../Model/OLD_1_encoder_3_decoders_complete_slices_single_channel_second_time/AE2_BEST.h5")
autoencoder_3.load_weights("../Model/OLD_1_encoder_3_decoders_complete_slices_single_channel_second_time/AE3_BEST.h5")

#print autoencoder_1.summary()

	

'''
encoder = Model(input_img, encoder_1(input_img))
encoder.compile(loss=root_mean_sq_GxGy, optimizer = SGD())
'''
psnr_gray_channel = []
psnr_gray_channel.append(1)
learning_rate = 0.000206
j=0
for jj in range(300):
	myfile_valid_psnr_7T = open('../1_encoder_3_decoders_complete_slices_single_channel_second_time/validation_psnr7T_1encoder_3decoders.txt', 'a')
	myfile_valid_mse_7T = open('../1_encoder_3_decoders_complete_slices_single_channel_second_time/validation_mse7T_1encoder_3decoders.txt', 'a')
	myfile_valid_psnr_dat_7T = open('../1_encoder_3_decoders_complete_slices_single_channel_second_time/validation_psnr7T_1encoder_3decoders.dat', 'a')
	myfile_valid_mse_dat_7T = open('../1_encoder_3_decoders_complete_slices_single_channel_second_time/validation_mse7T_1encoder_3decoders.dat', 'a')
	
	myfile_valid_psnr_3T = open('../1_encoder_3_decoders_complete_slices_single_channel_second_time/validation_psnr3T_1encoder_3decoders.txt', 'a')
	myfile_valid_mse_3T = open('../1_encoder_3_decoders_complete_slices_single_channel_second_time/validation_mse3T_1encoder_3decoders.txt', 'a')
	myfile_valid_psnr_dat_3T = open('../1_encoder_3_decoders_complete_slices_single_channel_second_time/validation_psnr3T_1encoder_3decoders.dat', 'a')
	myfile_valid_mse_dat_3T = open('../1_encoder_3_decoders_complete_slices_single_channel_second_time/validation_mse3T_1encoder_3decoders.dat', 'a')

	K.set_value(autoencoder_1.optimizer.lr, learning_rate)
	K.set_value(autoencoder_2.optimizer.lr, learning_rate)
	K.set_value(autoencoder_3.optimizer.lr, learning_rate)



	train_X,train_Y = shuffle(X_train,y_train)
	print ("Epoch is: %d\n" % j)
	print ("Number of batches: %d\n" % int(len(train_X)/batch_size))
	num_batches = int(len(train_X)/batch_size)
	for batch in range(num_batches):

		myfile_ae1_loss = open('../1_encoder_3_decoders_complete_slices_single_channel_second_time/ae1_train_loss_1encoder_3decoders.txt', 'a')
		myfile_ae2_loss = open('../1_encoder_3_decoders_complete_slices_single_channel_second_time/ae2_train_loss_1encoder_3decoders.txt', 'a')
		myfile_ae3_loss = open('../1_encoder_3_decoders_complete_slices_single_channel_second_time/ae3_train_loss_1encoder_3decoders.txt', 'a')
		myfile_dec1_loss = open('../1_encoder_3_decoders_complete_slices_single_channel_second_time/dec1_train_loss_1encoder_3decoders.txt', 'a')
		myfile_dec2_loss = open('../1_encoder_3_decoders_complete_slices_single_channel_second_time/dec2_train_loss_1encoder_3decoders.txt', 'a')
		myfile_dec3_loss = open('../1_encoder_3_decoders_complete_slices_single_channel_second_time/dec3_train_loss_1encoder_3decoders.txt', 'a')
		batch_train_X = train_X[batch*batch_size:min((batch+1)*batch_size,len(train_X)),:]
		batch_train_Y = train_Y[batch*batch_size:min((batch+1)*batch_size,len(train_Y)),:]
		loss_1 = autoencoder_1.test_on_batch(batch_train_X,batch_train_Y)
		loss_2 = autoencoder_2.test_on_batch(batch_train_X,batch_train_Y)
		loss_3 = autoencoder_3.test_on_batch(batch_train_X,batch_train_Y)
		print ('epoch_num: %d batch_num: %d Test_loss_1: %f\n' % (j,batch,loss_1))
		print ('epoch_num: %d batch_num: %d Test_loss_2: %f\n' % (j,batch,loss_2))
		print ('epoch_num: %d batch_num: %d Test_loss_3: %f\n' % (j,batch,loss_3))
		#myfile.write('epoch_num: %d batch_num: %d Test_loss_1: %f\n' % (j,batch,loss_1))
		#myfile.write('epoch_num: %d batch_num: %d Test_loss_1: %f\n' % (j,batch,loss_2))
		#myfile.write('epoch_num: %d batch_num: %d Test_loss_1: %f\n' % (j,batch,loss_3))
		if loss_1 < loss_2 and loss_1 < loss_3:
			train_1 = autoencoder_1.train_on_batch(batch_train_X,batch_train_Y)
			myfile_ae1_loss.write("%f \n" % (train_1))
			print ('epoch_num: %d batch_num: %d AE_Train_loss_1: %f\n' % (j,batch,train_1))
			#myfile.write('epoch_num: %d batch_num: %d AE_Train_loss_1: %f\n' % (j,batch,train_1))
			for layer in autoencoder_2.layers[:34]:
				layer.trainable = False
			for layer in autoencoder_3.layers[:34]:
				layer.trainable = False
			#autoencoder_2.summary()
			#autoencoder_3.summary()
			train_2 = autoencoder_2.train_on_batch(batch_train_X,batch_train_Y)
			train_3 = autoencoder_3.train_on_batch(batch_train_X,batch_train_Y)
			myfile_dec2_loss.write("%f \n" % (train_2))
			myfile_dec3_loss.write("%f \n" % (train_3))
			print ('epoch_num: %d batch_num: %d Decoder_Train_loss_2: %f\n' % (j,batch,train_2))
			print ('epoch_num: %d batch_num: %d Decoder_loss_3: %f\n' % (j,batch,train_3))
			#myfile.write('epoch_num: %d batch_num: %d Decoder_Train_loss_2: %f\n' % (j,batch,train_2))
			#myfile.write('epoch_num: %d batch_num: %d Decoder_loss_3: %f\n' % (j,batch,train_3))
		elif loss_2 < loss_1 and loss_2 < loss_3:
			train_2 = autoencoder_2.train_on_batch(batch_train_X,batch_train_Y)
			myfile_ae2_loss.write("%f \n" % (train_2))
			print ('epoch_num: %d batch_num: %d AE_Train_loss_2: %f\n' % (j,batch,train_2))
			#myfile.write('epoch_num: %d batch_num: %d AE_Train_loss_2: %f\n' % (j,batch,train_2))
			for layer in autoencoder_1.layers[:34]:
				layer.trainable = False
			for layer in autoencoder_3.layers[:34]:
				layer.trainable = False
			#autoencoder_1.summary()
			#autoencoder_3.summary()
			train_1 = autoencoder_1.train_on_batch(batch_train_X,batch_train_Y)
			train_3 = autoencoder_3.train_on_batch(batch_train_X,batch_train_Y)
			myfile_dec1_loss.write("%f \n" % (train_1))
			myfile_dec3_loss.write("%f \n" % (train_3))
			print ('epoch_num: %d batch_num: %d Decoder_Train_loss_1: %f\n' % (j,batch,train_1))
			print ('epoch_num: %d batch_num: %d Decoder_Train_loss_3: %f\n' % (j,batch,train_3))
			#myfile.write('epoch_num: %d batch_num: %d Decoder_Train_loss_1: %f\n' % (j,batch,train_1))
			#myfile.write('epoch_num: %d batch_num: %d Decoder_Train_loss_3: %f\n' % (j,batch,train_3))
		elif loss_3 < loss_1 and loss_3 < loss_2:
			train_3 = autoencoder_3.train_on_batch(batch_train_X,batch_train_Y)
			myfile_ae3_loss.write("%f \n" % (train_3))
			print ('epoch_num: %d batch_num: %d AE_Train_loss_3: %f\n' % (j,batch,train_3))
			#myfile.write('epoch_num: %d batch_num: %d AE_Train_loss_3: %f\n' % (j,batch,train_3))
			for layer in autoencoder_1.layers[:34]:
				layer.trainable = False
			for layer in autoencoder_2.layers[:34]:
				layer.trainable = False
			#autoencoder_1.summary()
			#autoencoder_2.summary()
			train_1 = autoencoder_1.train_on_batch(batch_train_X,batch_train_Y)
			train_2 = autoencoder_2.train_on_batch(batch_train_X,batch_train_Y)
			myfile_dec1_loss.write("%f \n" %(train_1))
			myfile_dec2_loss.write("%f \n" % (train_2))
			print ('epoch_num: %d batch_num: %d Decoder_Train_loss_1: %f\n' % (j,batch,train_1))
			print ('epoch_num: %d batch_num: %d Decoder_Train_loss_2: %f\n' % (j,batch,train_2))
			#myfile.write('epoch_num: %d batch_num: %d Decoder_Train_loss_1: %f\n' % (j,batch,train_1))
			#myfile.write('epoch_num: %d batch_num: %d Decoder_Train_loss_2: %f\n' % (j,batch,train_2))
		elif loss_1 == loss_2:
			train_1 = autoencoder_1.train_on_batch(batch_train_X,batch_train_Y)
			myfile_ae1_loss.write("%f \n" % (train_1))
			print ('epoch_num: %d batch_num: %d AE_Train_loss_1_equal_state: %f\n' % (j,batch,train_1))
			#myfile.write('epoch_num: %d batch_num: %d AE_Train_loss_1_equal_state: %f\n' % (j,batch,train_1))
			for layer in autoencoder_3.layers[:34]:
				layer.trainable = False
			for layer in autoencoder_2.layers[:34]:
				layer.trainable = False
			#autoencoder_2.summary()
			#autoencoder_3.summary()
			train_2 = autoencoder_2.train_on_batch(batch_train_X,batch_train_Y)
			train_3 = autoencoder_3.train_on_batch(batch_train_X,batch_train_Y)
			myfile_dec2_loss.write("%f \n" % (train_2))
			myfile_dec3_loss.write("%f \n" % (train_3))
			print ('epoch_num: %d batch_num: %d Decoder_Train_loss_2_equal_state: %f\n' % (j,batch,train_2))
			print ('epoch_num: %d batch_num: %d Decoder_Train_loss_3_equal_state: %f\n' % (j,batch,train_3))
			#myfile.write('epoch_num: %d batch_num: %d Decoder_Train_loss_2_equal_state: %f\n' % (j,batch,train_2))
			#myfile.write('epoch_num: %d batch_num: %d Decoder_Train_loss_3_equal_state: %f\n' % (j,batch,train_3))
		elif loss_2 == loss_3: 
			train_2 = autoencoder_2.train_on_batch(batch_train_X,batch_train_Y)
			myfile_ae2_loss.write("%f \n" % (train_2))
			print ('epoch_num: %d batch_num: %d AE_Train_loss_2_equal_state: %f\n' % (j,batch,train_2))
			#myfile.write('epoch_num: %d batch_num: %d AE_Train_loss_2_equal_state: %f\n' % (j,batch,train_2))
			for layer in autoencoder_1.layers[:34]:
				layer.trainable = False
			for layer in autoencoder_3.layers[:34]:
				layer.trainable = False

			#autoencoder_2.summary()
			#autoencoder_3.summary()
			train_1 = autoencoder_1.train_on_batch(batch_train_X,batch_train_Y)
			train_3 = autoencoder_3.train_on_batch(batch_train_X,batch_train_Y)
			myfile_dec1_loss.write("%f \n" % (train_1))
			myfile_dec3_loss.write("%f \n" % (train_3))
			print ('epoch_num: %d batch_num: %d Decoder_Train_loss_1_equal_state: %f\n' % (j,batch,train_1))
			print ('epoch_num: %d batch_num: %d Decoder_Train_loss_3_equal_state: %f\n' % (j,batch,train_3))
			#myfile.write('epoch_num: %d batch_num: %d Decoder_Train_loss_1_equal_state: %f\n' % (j,batch,train_1))
			#myfile.write('epoch_num: %d batch_num: %d Decoder_Train_loss_3_equal_state: %f\n' % (j,batch,train_3))
		elif loss_3 == loss_1:
			train_1 = autoencoder_1.train_on_batch(batch_train_X,batch_train_Y)
			myfile_ae1_loss.write("%f \n" % (train_1))
			print ('epoch_num: %d batch_num: %d AE_Train_loss_1_equal_state: %f\n' % (j,batch,train_1))
			#myfile.write('epoch_num: %d batch_num: %d AE_Train_loss_1_equal_state: %f\n' % (j,batch,train_1))
			for layer in autoencoder_2.layers[:34]:
				layer.trainable = False
			for layer in autoencoder_3.layers[:34]:
				layer.trainable = False
			#autoencoder_2.summary()
			#autoencoder_3.summary()
			train_2 = autoencoder_2.train_on_batch(batch_train_X,batch_train_Y)
			train_3 = autoencoder_3.train_on_batch(batch_train_X,batch_train_Y)
			myfile_dec2_loss.write("%f \n" % (train_2))
			myfile_dec3_loss.write("%f \n" % (train_3))
			print ('epoch_num: %d batch_num: %d Decoder_Train_loss_2_equal_state: %f\n' % (j,batch,train_2))
			print ('epoch_num: %d batch_num: %d Decoder_Train_loss_3_equal_state: %f\n' % (j,batch,train_3))
			#myfile.write('epoch_num: %d batch_num: %d Decoder_Train_loss_2_equal_state: %f\n' % (j,batch,train_2))
			#myfile.write('epoch_num: %d batch_num: %d Decoder_Train_loss_3_equal_state: %f\n' % (j,batch,train_3))
	

			myfile_ae1_loss.close()
			myfile_ae2_loss.close()
			myfile_ae3_loss.close()
			myfile_dec1_loss.close()
			myfile_dec2_loss.close()
			myfile_dec3_loss.close()
	for layer in autoencoder_1.layers[:34]:
				layer.trainable = True
	for layer in autoencoder_2.layers[:34]:
				layer.trainable = True
	for layer in autoencoder_3.layers[:34]:
				layer.trainable = True

	if jj % 100 ==0:
			autoencoder_1.save_weights("../Model/1_encoder_3_decoders_complete_slices_single_channel_second_time/ae1_1_encoder_3_decoders_complete_slices_min_single_channel_second_time_" + str(jj)+".h5")
			autoencoder_2.save_weights("../Model/1_encoder_3_decoders_complete_slices_single_channel_second_time/ae2_1_encoder_3_decoders_complete_slices_min_single_channel_second_time_" + str(jj)+".h5")
			autoencoder_3.save_weights("../Model/1_encoder_3_decoders_complete_slices_single_channel_second_time/ae3_1_encoder_3_decoders_complete_slices_min_single_channel_second_time_" + str(jj)+".h5")


	autoencoder_1.save_weights("../Model/1_encoder_3_decoders_complete_slices_single_channel_second_time/ae1_1_encoder_3_decoders_complete_slices_min_single_channel_second_time.h5")
	autoencoder_2.save_weights("../Model/1_encoder_3_decoders_complete_slices_single_channel_second_time/ae2_1_encoder_3_decoders_complete_slices_min_single_channel_second_time.h5")
	autoencoder_3.save_weights("../Model/1_encoder_3_decoders_complete_slices_single_channel_second_time/ae3_1_encoder_3_decoders_complete_slices_min_single_channel_second_time.h5")
	
	#autoencoder.save_weights("../Model/1_slices51_Masked_MSE.h5")
	X_test,y_test = shuffle(X_test,y_test)
	if loss_1 < loss_2 and loss_1 < loss_3:
		decoded_imgs = autoencoder_1.predict(X_test)
		mse_7T=  np.mean((y_test[:,:,:,0] - decoded_imgs[:,:,:,0]) ** 2)
		check_7T = math.sqrt(mse_7T)
		psnr_7T = 20 * math.log10( 1.0 / check_7T)
		mse_3T=  np.mean((X_test[:,:,:,0] - decoded_imgs[:,:,:,0]) ** 2)
		check_3T = math.sqrt(mse_3T)
		psnr_3T = 20 * math.log10( 1.0 / check_3T)

		myfile_valid_psnr_7T.write("%f \n" % (psnr_7T))
		myfile_valid_mse_7T.write("%f \n" % (mse_7T))
		myfile_valid_psnr_dat_7T.write("%f \n" % (psnr_7T))
		myfile_valid_mse_dat_7T.write("%f \n" % (mse_7T))
		
		myfile_valid_psnr_3T.write("%f \n" % (psnr_3T))
		myfile_valid_mse_3T.write("%f \n" % (mse_3T))
		myfile_valid_psnr_dat_3T.write("%f \n" % (psnr_3T))
		myfile_valid_mse_dat_3T.write("%f \n" % (mse_3T))
		#print (check)
	elif loss_2 < loss_1 and loss_2 < loss_3:
		decoded_imgs = autoencoder_2.predict(X_test)
		mse_7T=  np.mean((y_test[:,:,:,0] - decoded_imgs[:,:,:,0]) ** 2)
		check_7T = math.sqrt(mse_7T)
		psnr_7T = 20 * math.log10( 1.0 / check_7T)
		mse_3T=  np.mean((X_test[:,:,:,0] - decoded_imgs[:,:,:,0]) ** 2)
		check_3T = math.sqrt(mse_3T)

		psnr_3T = 20 * math.log10( 1.0 / check_3T)
		myfile_valid_psnr_7T.write("%f \n" % (psnr_7T))
		myfile_valid_mse_7T.write("%f \n" % (mse_7T))
		myfile_valid_psnr_dat_7T.write("%f \n" % (psnr_7T))
		myfile_valid_mse_dat_7T.write("%f \n" % (mse_7T))

		myfile_valid_psnr_3T.write("%f \n" % (psnr_3T))
		myfile_valid_mse_3T.write("%f \n" % (mse_3T))
		myfile_valid_psnr_dat_3T.write("%f \n" % (psnr_3T))
		myfile_valid_mse_dat_3T.write("%f \n" % (mse_3T))
		#print (check)
	elif loss_3 < loss_2 and loss_3 < loss_1:
		decoded_imgs = autoencoder_3.predict(X_test)
		mse_7T=  np.mean((y_test[:,:,:,0] - decoded_imgs[:,:,:,0]) ** 2)
		check_7T = math.sqrt(mse_7T)
		psnr_7T = 20 * math.log10( 1.0 / check_7T)
		mse_3T=  np.mean((X_test[:,:,:,0] - decoded_imgs[:,:,:,0]) ** 2)
		check_3T = math.sqrt(mse_3T)
		psnr_3T = 20 * math.log10( 1.0 / check_3T)
	
		myfile_valid_psnr_7T.write("%f \n" % (psnr_7T))
		myfile_valid_mse_7T.write("%f \n" % (mse_7T))
		myfile_valid_psnr_dat_7T.write("%f \n" % (psnr_7T))
		myfile_valid_mse_dat_7T.write("%f \n" % (mse_7T))

		myfile_valid_psnr_3T.write("%f \n" % (psnr_3T))
		myfile_valid_mse_3T.write("%f \n" % (mse_3T))
		myfile_valid_psnr_dat_3T.write("%f \n" % (psnr_3T))
		myfile_valid_mse_dat_3T.write("%f \n" % (mse_3T))
		#print (check)
	elif loss_1 == loss_2:
		decoded_imgs = autoencoder_1.predict(X_test)
		mse_7T=  np.mean((y_test[:,:,:,0] - decoded_imgs[:,:,:,0]) ** 2)
		check_7T = math.sqrt(mse_7T)
		psnr_7T = 20 * math.log10( 1.0 / check_7T)
		mse_3T=  np.mean((X_test[:,:,:,0] - decoded_imgs[:,:,:,0]) ** 2)
		check_3T = math.sqrt(mse_3T)
		psnr_3T = 20 * math.log10( 1.0 / check_3T)

		myfile_valid_psnr_7T.write("%f \n" % (psnr_7T))
		myfile_valid_mse_7T.write("%f \n" % (mse_7T))
		myfile_valid_psnr_dat_7T.write("%f \n" % (psnr_7T))
		myfile_valid_mse_dat_7T.write("%f \n" % (mse_7T))

		myfile_valid_psnr_3T.write("%f \n" % (psnr_3T))
		myfile_valid_mse_3T.write("%f \n" % (mse_3T))
		myfile_valid_psnr_dat_3T.write("%f \n" % (psnr_3T))
		myfile_valid_mse_dat_3T.write("%f \n" % (mse_3T))
		
	elif loss_2 == loss_3:
		decoded_imgs = autoencoder_2.predict(X_test)
		mse_7T=  np.mean((y_test[:,:,:,0] - decoded_imgs[:,:,:,0]) ** 2)
		check_7T = math.sqrt(mse_7T)
		psnr_7T = 20 * math.log10( 1.0 / check_7T)
		mse_3T=  np.mean((X_test[:,:,:,0] - decoded_imgs[:,:,:,0]) ** 2)
		check_3T = math.sqrt(mse_3T)
		psnr_3T = 20 * math.log10( 1.0 / check_3T)

		myfile_valid_psnr_7T.write("%f \n" % (psnr_7T))
		myfile_valid_mse_7T.write("%f \n" % (mse_7T))
		myfile_valid_psnr_dat_7T.write("%f \n" % (psnr_7T))
		myfile_valid_mse_dat_7T.write("%f \n" % (mse_7T))

		myfile_valid_psnr_3T.write("%f \n" % (psnr_3T))
		myfile_valid_mse_3T.write("%f \n" % (mse_3T))
		myfile_valid_psnr_dat_3T.write("%f \n" % (psnr_3T))
		myfile_valid_mse_dat_3T.write("%f \n" % (mse_3T))		

	elif loss_3 == loss_1:
		decoded_imgs = autoencoder_3.predict(X_test)
		mse_7T=  np.mean((y_test[:,:,:,0] - decoded_imgs[:,:,:,0]) ** 2)
		check_7T = math.sqrt(mse_7T)
		psnr_7T = 20 * math.log10( 1.0 / check_7T)
		mse_3T=  np.mean((X_test[:,:,:,0] - decoded_imgs[:,:,:,0]) ** 2)
		check_3T = math.sqrt(mse_3T)
		psnr_3T = 20 * math.log10( 1.0 / check_3T)

		myfile_valid_psnr_7T.write("%f \n" % (psnr_7T))
		myfile_valid_mse_7T.write("%f \n" % (mse_7T))
		myfile_valid_psnr_dat_7T.write("%f \n" % (psnr_7T))
		myfile_valid_mse_dat_7T.write("%f \n" % (mse_7T))

		myfile_valid_psnr_3T.write("%f \n" % (psnr_3T))
		myfile_valid_mse_3T.write("%f \n" % (mse_3T))
		myfile_valid_psnr_dat_3T.write("%f \n" % (psnr_3T))
		myfile_valid_mse_dat_3T.write("%f \n" % (mse_3T))


	if max(psnr_gray_channel) < psnr_7T:
			autoencoder_1.save_weights("../Model/1_encoder_3_decoders_complete_slices_single_channel_second_time/			     BEST_ae1_1_encoder_3_decoders_complete_slices_min_single_channel_second_time_" + str(jj)+".h5")
			autoencoder_2.save_weights("../Model/1_encoder_3_decoders_complete_slices_single_channel_second_time/BEST_ae2_1_encoder_3_decoders_complete_slices_min_single_channel_second_time_" + str(jj)+".h5")
			autoencoder_3.save_weights("../Model/1_encoder_3_decoders_complete_slices_single_channel_second_time/BEST_ae3_1_encoder_3_decoders_complete_slices_min_single_channel_second_time_" + str(jj)+".h5")

	psnr_gray_channel.append(psnr_7T)

		
	temp = np.zeros([resizeTo,resizeTo*3])
	temp[:resizeTo,:resizeTo] = X_test[0,:,:,0]
	#temp[resizeTo:resizeTo*2,:resizeTo] = X_test[0,:,:,1]
	#temp[resizeTo*2:,:resizeTo] = X_test[0,:,:,2]
	temp[:resizeTo,resizeTo:resizeTo*2] = y_test[0,:,:,0]
	#temp[resizeTo:resizeTo*2,resizeTo:resizeTo*2] = y_test[0,:,:,1]
	#temp[resizeTo*2:,resizeTo:resizeTo*2] = y_test[0,:,:,2]
	temp[:resizeTo,2*resizeTo:] = decoded_imgs[0,:,:,0]
	#temp[resizeTo:resizeTo*2,2*resizeTo:] = decoded_imgs[0,:,:,1]
	#temp[resizeTo*2:,2*resizeTo:] = decoded_imgs[0,:,:,2]
	temp = temp*255
	scipy.misc.imsave('../AllResults/1_encoder_3_decoders_complete_slices_single_channel_second_time/' + str(j) + '.jpg', temp)
	j +=1
	myfile_valid_psnr_7T.close()
	myfile_valid_mse_7T.close()
	myfile_valid_psnr_dat_7T.close()
	myfile_valid_mse_dat_7T.close()

	myfile_valid_psnr_3T.close()
	myfile_valid_mse_3T.close()
	myfile_valid_psnr_dat_3T.close()
	myfile_valid_mse_dat_3T.close()

	if jj % 20 ==0:
		learning_rate = learning_rate - learning_rate * 0.10

