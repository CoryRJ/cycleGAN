import tensorflow as tf
import sys
import numpy as np
import matplotlib.pyplot as plt
from time import time
from datetime import timedelta
import batch_class as bc
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

img_dims = (135,240,3)

col = 6
row = 6
state_vars = 'variables.txt'

total_runs = 30000000000
lr = 0.000002
loop_for = 100
batch_size = 5
grab_sample = 100
get_out = 'n'

learning_r = tf.placeholder(tf.float32,[])
A_in_img = tf.placeholder(tf.float32,[None,img_dims[0],img_dims[1],3])
B_in_img = tf.placeholder(tf.float32,[None,img_dims[0],img_dims[1],3])

def update_globals(state_vars):
	global lr
	global loop_for
	global batch_size
	global get_out
	try:
		with open(state_vars) as f:
			content = f.readlines()
		content = [x.strip() for x in content]
		lr = float(content[0])
		loop_for =int(content[1])
		batch_size =int(content[2])
		get_out =str(content[3])
	except:
		print('Could not update all variables.')
	print('Learning rate is: ', lr)
	print('Loop for is: ', loop_for)
	print('Batch size is: ', batch_size)
	print('Get out is: ', get_out)

	

def conv_res_block(x,ks,act,fil):
	res = tf.layers.conv2d(inputs=x,filters=fil,kernel_size=ks,padding='same')
	res = tf.layers.batch_normalization(res)
	res = act(res)
	res = tf.layers.conv2d(inputs=res,filters=fil,kernel_size=ks,padding='same')
	res = tf.layers.batch_normalization(res) + x
	res = act(res)
	return res


def disc_base(data):
	rel = tf.nn.relu
	lrel = tf.nn.leaky_relu
	ks = 4
	pad='same'
	stride =2
	
	dis = tf.layers.conv2d(inputs=data,filters=64,kernel_size=ks,strides =stride,padding=pad)
	dis = tf.layers.batch_normalization(dis)
	dis = lrel(dis)
	dis = tf.layers.conv2d(inputs=dis,filters=128,kernel_size=ks,strides =stride,padding=pad)
	dis = tf.layers.batch_normalization(dis)
	dis = lrel(dis)
	dis = tf.layers.conv2d(inputs=dis,filters=256,kernel_size=ks,strides =stride,padding=pad)
	dis = tf.layers.batch_normalization(dis)
	dis = lrel(dis)
	dis = tf.layers.conv2d(inputs=dis,filters=512,kernel_size=ks,strides =1,padding=pad)
	dis = tf.layers.batch_normalization(dis)
	dis = lrel(dis)
	dis = tf.layers.conv2d(inputs=dis,filters=1,kernel_size=ks,strides =1,padding=pad)
	dis = tf.layers.batch_normalization(dis)
	dis = lrel(dis)
	print(dis.get_shape())
	dis = tf.reshape(dis,[-1,17*30]) #currently, this is hard coded in. Make sure to change if you change the size of images
	dis = tf.layers.dense(dis,1)
	out = tf.nn.sigmoid(dis)
	#out = tf.math.reduce_mean(out,axis = [1,2,3])

	return out


def gen_base(data):
	rel = tf.nn.relu
	lrel = tf.nn.leaky_relu
	ks = 4
	out_lay =256
	stride = 2

	data = tf.image.resize_image_with_crop_or_pad(data,136,240)
	enc1 = tf.layers.conv2d(inputs=data,filters=64,kernel_size=7,strides =1,padding='same')
	enc1 = lrel(enc1)

	enc2 = tf.layers.conv2d(inputs=enc1,filters=128,kernel_size=ks,strides=stride,padding='same')
	enc2 = tf.layers.batch_normalization(enc2)
	enc2 = lrel(enc2)

	enc3 = tf.layers.conv2d(inputs=enc2,filters=out_lay,kernel_size=ks,strides=stride,padding='same')
	enc3 = tf.layers.batch_normalization(enc3)
	enc3 = lrel(enc3)


	res = conv_res_block(enc3,ks,lrel,out_lay)
	res = conv_res_block(res,ks,lrel,out_lay)
	res = conv_res_block(res,ks,lrel,out_lay)

	res = conv_res_block(res,ks,lrel,out_lay)
	res = conv_res_block(res,ks,lrel,out_lay)
	res = conv_res_block(res,ks,lrel,out_lay)
	#res = tf.concat((res,enc3),3) #SKIP


	print(res.get_shape())
	dec = tf.layers.conv2d_transpose(inputs=res,filters=128,kernel_size=ks,strides =stride,padding='same')
	dec = tf.layers.batch_normalization(dec)
	dec = dec+enc2 #SKIP
	dec = lrel(dec)


	print(dec.get_shape())
	dec = tf.layers.conv2d_transpose(inputs=dec,filters=64,kernel_size=3,strides =stride,padding='same')
	dec = tf.layers.batch_normalization(dec)
	dec = dec+enc1 #SKIP
	dec = lrel(dec)

	dec = tf.layers.conv2d_transpose(inputs=dec,filters=64,kernel_size=4,strides =1,padding='same')
	dec = tf.layers.batch_normalization(dec)
	dec = lrel(dec)


	print(dec.get_shape())
	dec = tf.layers.conv2d_transpose(inputs=dec,filters=3,kernel_size=7,strides =1,padding='same')
	dec = rel(dec)

	out = dec
	out = tf.image.resize_image_with_crop_or_pad(out,135,240)
	return out
def sce_cost(lab, log):
	return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=lab,logits=log))
def mse_cost(lab, log):
	return tf.reduce_mean(tf.losses.mean_squared_error(labels=lab,predictions=log))

def getRand(x,y):
	return np.random.uniform(-1.0,1.0,(x,y))
	#return np.random.randn(x,y)

def train_neural_network():
	global lr
	global get_out
	num_of_img = 84048
	path = 'path_to_class_A'
	start=0
	img_type ='.png'
	class_A = bc.Batch(num_of_img,path,img_dims,batch_size,start,img_type)

	num_of_img = 97919
	path = 'path_to_class_B'
	start=1
	img_type ='.png'
	class_B = bc.Batch(num_of_img,path,img_dims,batch_size,start,img_type)

	fig,_ = plt.subplots(col, row,num ='Images')
	fig.suptitle('Images')

	cst = plt.figure('Total Cost')
	cst.suptitle('Total Cost')
	#Generators: Read as A to B or B to A. Others follow this pattern as well.
	with tf.variable_scope('g/A_B'):
		A_B = gen_base(A_in_img)
	with tf.variable_scope('g/B_A'):
		B_A = gen_base(B_in_img)

	#ENCODER DECODER pairs
	with tf.variable_scope('g/A_B', reuse = True):
		B_A_B = gen_base(B_A)
	with tf.variable_scope('g/B_A', reuse = True):
		A_B_A = gen_base(A_B)


	#Discrimnator for males
	with tf.variable_scope('d/A'):
		A_d_fake = disc_base(B_A)
	with tf.variable_scope('d/A', reuse = True):
		A_d_real = disc_base(A_in_img)

	#Discriminator for females
	with tf.variable_scope('d/B'):
		B_d_fake = disc_base(A_B)
	with tf.variable_scope('d/B', reuse = True):
		B_d_real = disc_base(B_in_img)


	A_d_fake_cost = mse_cost(tf.zeros_like(A_d_fake),A_d_fake)
	A_d_real_cost = mse_cost( tf.ones_like(A_d_real)*.9,A_d_real)
	A_d_total_cost = tf.add(A_d_fake_cost,A_d_real_cost)

	B_d_fake_cost = mse_cost(tf.zeros_like(B_d_fake),B_d_fake)
	B_d_real_cost = mse_cost( tf.ones_like(B_d_real)*.9,B_d_real)
	B_d_total_cost = tf.add(B_d_fake_cost,B_d_real_cost)

	B_A_cost = mse_cost(tf.ones_like(A_d_fake)*.9,A_d_fake)
	A_B_cost = mse_cost(tf.ones_like(B_d_fake)*.9,B_d_fake)

	A_B_A_cost = mse_cost(A_in_img,A_B_A)
	B_A_B_cost = mse_cost(B_in_img,B_A_B)

	gen_cost =  A_B_cost + B_A_cost + (A_B_A_cost + B_A_B_cost)*10
	dis_cost = (A_d_total_cost + B_d_total_cost)

	adam = tf.train.AdamOptimizer(learning_rate=learning_r,beta1=0.5)

	variables_to_train =tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'g/')
	gen_opt = adam.minimize(gen_cost,var_list=variables_to_train)

	variables_to_train =tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'d/')
	dis_opt = adam.minimize(dis_cost,var_list=variables_to_train)

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	saver = tf.train.Saver()
	running_time = time()
	g_cost = []
	d_cost = []
	with tf.Session(config = config) as sess:
		sess.run(tf.global_variables_initializer())
		#saver.restore(sess,".\\saves\\test_one\\test.ckpt"
		dt_c = 0
		gr_c = 0
		dt_total_loss = 0
		gr_total_loss = 0
		dt_dif = 0
		gr_dif = 0

		time_takes = 0
		training_steps = 0
		for update_steps in range(total_runs):
			dt_total_loss = 0
			gr_total_loss = 0
			for i in range(loop_for):
				training_steps += 1
				start = time()
				dt_c,gr_c,_,_=sess.run([dis_cost,gen_cost, dis_opt,gen_opt],feed_dict={
														learning_r:lr,
														A_in_img:class_A.get_batch_random(batch_size),
														B_in_img:class_B.get_batch_random(batch_size)
														})
				time_takes += (time() - start)/batch_size
				start = time()
				dt_total_loss += dt_c
				gr_total_loss += gr_c
				dt_dif += dt_c
				gr_dif += gr_c

				if(i%(loop_for/10) ==0):
					seconds = time_takes/training_steps
					print("Dis Loss: ", dt_c, " Reconstruct Loss: ", gr_c)
					print("One training batch took on average: ", str(timedelta(seconds=int(seconds*batch_size))),' h:m:s. Estimated remaining time is: ',str(timedelta(seconds=int((loop_for-i)*seconds*batch_size))),' h:m:s.' )
					print("Completed: ", i, "/", loop_for,".")
				if(training_steps%grab_sample == grab_sample-1):
					g_cost = np.concatenate((g_cost,[gr_dif/grab_sample]),0)
					d_cost = np.concatenate((d_cost,[dt_dif/grab_sample]),0)
					dt_dif = 0
					gr_dif = 0

				
				plt.pause(0.001)
				time_takes += (time() - start)
			if(update_steps%1 == 0):
				get_out = 'n'
				while(get_out != 'y'):
					from_A =class_A.get_batch_random(6)
					from_B = class_B.get_batch_random(6)
					fake_A,fake_B,rev_B,rev_A = sess.run([B_A,A_B,B_A_B,A_B_A],feed_dict={
											A_in_img:from_A,
											B_in_img:from_B
											})


					visual_show = np.concatenate((from_A,rev_A),0)
					visual_show = np.concatenate((visual_show,fake_B),0)
					visual_show = np.concatenate((visual_show,from_B),0)
					visual_show = np.concatenate((visual_show,rev_B),0)
					visual_show = np.concatenate((visual_show,fake_A),0)

					plt.figure('Images')
					plt.clf()
					for p in range(row*col):
						fig.add_subplot(row,col,p+1)
					for ax,p in zip(fig.axes,visual_show):
						ax.imshow(p)
					plt.show(block=False)
					plt.pause(0.001)

					plt.figure('Total Cost')
					plt.clf()
					plt.plot(g_cost,'g-')
					plt.plot(d_cost,'b-')
					plt.show(block=False)
					plt.pause(0.001)

					visual_show = sess.run([A_d_fake,A_d_real,B_d_fake,B_d_real],feed_dict={
											A_in_img:from_A,
											B_in_img:from_B
											})
					print('A_d_fake')
					for i in visual_show[0]:
						print(i,end =' ')
					print()
					print('A_d_real')
					for i in visual_show[1]:
						print(i,end =' ')
					print()
					print('B_d_fake')
					for i in visual_show[2]:
						print(i,end =' ')
					print()
					print('B_d_real')
					for i in visual_show[3]:
						print(i,end =' ')
					print()
					update_globals(state_vars)
					if(get_out != 'y'):
						get_out = input("type y to leave: ")
				save_path = saver.save(sess,".\\saves\\test_m1\\test.ckpt")
				print("Saved at: ",save_path)
			print()
			print ('Update Steps: ', update_steps+1, ' completed out of ', total_runs, ' Total Dis Loss: ', dt_total_loss/loop_for, ' Reconstruct Total Loss: ', gr_total_loss/loop_for)
			print('Time Running:', str(timedelta(seconds=int(time()-running_time))))



update_globals(state_vars)
train_neural_network()
	
