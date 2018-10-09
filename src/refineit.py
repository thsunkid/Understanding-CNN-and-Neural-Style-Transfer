import os
import numpy as np
from scipy.misc import imsave, toimage
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tensorflow as tf
import cv2
import PIL.Image
from functools import partial
from naive import display_layers, animationfig, deprocess_image, show_save

model_fn = '~/tensorflow_inception_graph.pb'
arg = [
	('mixed4d_3x3_bottleneck_pre_relu',139), # 0. Flower 
	('mixed4a',11) , # 1. neuron network
	('head0_bottleneck_pre_relu', 53), # 2. Feathers
	('mixed5a_3x3_bottleneck_pre_relu', 11), # 3. Dog face and circles
	('mixed5a_3x3_bottleneck_pre_relu', 119), # 4. Butterfly
	('mixed5a_3x3_bottleneck_pre_relu', 33), # 5. Spider monkey brains
	('mixed4a_3x3_bottleneck_pre_relu',35), # 6. Sparkle
	('head0_bottleneck_pre_relu', 18), # 7. Trumpets
	('head0_bottleneck_pre_relu', 101), # 8. dog face
	('head0_bottleneck_pre_relu', 88), # 9. dog face again
	('head0_bottleneck_pre_relu', 23), # 10. Eye waves
	('head0_bottleneck_pre_relu', 26), # 11. Network
	('head0_bottleneck_pre_relu', 47), # 12. Pyramids
	('mixed4a_3x3_bottleneck_pre_relu', 42), # 13. worms
	('mixed4b_3x3_bottleneck_pre_relu', 68), # 14. fur  
	('mixed4b_pool_reduce_pre_relu',16), #15. Flower again!
]

### config
unit = 8
_save = False
_gamma = False
_save_name = 'hoa1'


### Graph model
graph = tf.Graph()
sess = tf.InteractiveSession(graph=graph)
with tf.gfile.FastGFile(os.path.expanduser(model_fn), 'rb') as f:
	graph_def = tf.GraphDef()
	graph_def.ParseFromString(f.read())



t_input = tf.placeholder(np.float32, name='input') # define the input tensor
imagenet_mean = 117.0 # define wrt imagenet's standard
t_preprocessed = tf.expand_dims(t_input-imagenet_mean, 0)
tf.import_graph_def(graph_def, {'input':t_preprocessed})

layers = [op.name for op in graph.get_operations() if op.type=='Conv2D' and 'import/' in op.name]
feature_nums = [int(graph.get_tensor_by_name(name+':0').get_shape()[-1]) for name in layers]



def tffunc(*argtypes):
	'''Helper that transforms TF-graph generating function into a regular one.
	See "resize" function below.
	'''
	placeholders = list(map(tf.placeholder, argtypes))
	def wrap(f):
		out = f(*placeholders)
		def wrapper(*args, **kw):
			return out.eval(dict(zip(placeholders, args)), session=kw.get('session'))
		return wrapper
	return wrap


# Helper function that uses TF to resize an image
def resize(img, size):
	img = tf.expand_dims(img, 0)
	return tf.image.resize_bilinear(img, size)[0,:,:,:]
resize = tffunc(np.float32, np.int32)(resize)

def T(layer):
	'''Helper for getting layer output tensor'''
	return graph.get_tensor_by_name("import/%s:0"%layer)

def calc_grad_tiled(img, t_grad, t_score, tile_size=512):
	'''Compute the value of tensor t_grad over the image in a tiled way.
	Random shifts are applied to the image to blur tile boundaries over 
	multiple iterations.'''
	sz = tile_size
	h, w = img.shape[:2]
	sx, sy = np.random.randint(sz, size=2)
	img_shift = np.roll(np.roll(img, sx, 1), sy, 0)
	grad = np.zeros_like(img)
	for y in range(0, max(h-sz//2, sz),sz):
		for x in range(0, max(w-sz//2, sz),sz):
			sub = img_shift[y:y+sz,x:x+sz]
			g,score = sess.run([t_grad, t_score], {t_input:sub})
			grad[y:y+sz,x:x+sz] = g
	return np.roll(np.roll(grad, -sx, 1), -sy, 0), score

def render_multiscale(t_obj, img0, iter_n=10, step=1.0, octave_n=3, octave_scale=1.4,name='tensorflower1'):
	t_score = tf.reduce_mean(t_obj)  
	t_grad = tf.gradients(t_score, t_input)[0]  
	
	img = img0.copy()
	imagelist.append(deprocess_image(img))
	for octave in range(octave_n):
		if octave>0:
			hw = np.float32(img.shape[:2])*octave_scale
			# show_save(img)
			print("before scale:",img.shape)
			img = resize(img, np.int32(hw))
			print("after scale:",img.shape)
   
		for i in range(iter_n):
			g, score = sess.run([t_grad, t_score], {t_input:img})
			# g,score = calc_grad_tiled(img, t_grad,t_score)

			# normalizing the gradient, so the same step size should work 
			g /= g.std()+1e-8         
			img += g*step
			imagelist.append(deprocess_image(img)) 
			print("Step ",i,", Loss value:",score)
	show_save(img,_save,_gamma,name=name)
	return img 

k = np.float32([1,4,6,4,1])
k = np.outer(k, k)
k5x5 = k[:,:,None,None]/k.sum()*np.eye(3, dtype=np.float32)

# Laplacian Pyramid Gradient 

def lap_split(img):
	'''Split the image into lo and hi frequency components'''
	with tf.name_scope('split'):
		lo = tf.nn.conv2d(img, k5x5, [1,2,2,1], 'SAME')
		lo2 = tf.nn.conv2d_transpose(lo, k5x5*4, tf.shape(img), [1,2,2,1])
		hi = img-lo2
	return lo, hi

def lap_split_n(img, n):
	'''Build Laplacian pyramid with n splits'''
	levels = []
	for i in range(n):
		img, hi = lap_split(img)
		levels.append(hi)
	levels.append(img)
	return levels[::-1]

def lap_merge(levels):
	'''Merge Laplacian pyramid'''
	img = levels[0]
	for hi in levels[1:]:
		with tf.name_scope('merge'):
			img = tf.nn.conv2d_transpose(img, k5x5*4, tf.shape(hi), [1,2,2,1]) + hi
	return img

def normalize_std(img, eps=1e-10):
	'''Normalize image by making its standard deviation = 1.0'''
	with tf.name_scope('normalize'):
		std = tf.sqrt(tf.reduce_mean(tf.square(img)))
		return img/tf.maximum(std, eps)


def lap_normalize(img, scale_n=4):
	'''Perform the Laplacian pyramid normalization.'''
	img = tf.expand_dims(img,0)
	tlevels = lap_split_n(img, scale_n)
	tlevels = list(map(normalize_std, tlevels))
	out = lap_merge(tlevels)
	return out[0,:,:,:]

# Gradient Laplacian Pyramid normalization is a kind 
# of adaptive learning rate approach in the same space.

def render_lapnorm(t_obj, img0,
				   iter_n=10, step=1.0, octave_n=3, octave_scale=1.4, lap_n=4):
	t_score = tf.reduce_mean(t_obj) 
	t_grad = tf.gradients(t_score, t_input)[0]  
	# build the laplacian normalization graph
	lap_norm_func = tffunc(np.float32)(partial(lap_normalize, scale_n=lap_n))

	img = img0.copy()
	for octave in range(octave_n):
		if octave>0:
			hw = np.float32(img.shape[:2])*octave_scale
			img = resize(img, np.int32(hw))
	
		for i in range(iter_n):
			# g,score = calc_grad_tiled(img, t_grad,t_score)
			g,score =sess.run([t_grad, t_score], {t_input:img})

			g = lap_norm_func(g)
			img += g*step
			print("Step ",i,", Loss value:",score)
 
	show_save(img,save = _save,gamma = _gamma, name = _save_name)

if __name__ == '__main__':
	img_noise = np.random.uniform(size=(224,224,3)) + 100.0
	imagelist = [] # for animation
	obj = T(arg[unit][0])[:,:,:,arg[unit][1]]
	# _img = render_multiscale(obj,img_noise,octave_n=3)
	# animationfig(imagelist)

	render_lapnorm(obj, img_noise, octave_n=3)
