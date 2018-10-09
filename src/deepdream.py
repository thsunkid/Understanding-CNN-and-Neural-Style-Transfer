import os
import numpy as np
from scipy.misc import imsave, toimage
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tensorflow as tf
import cv2
from functools import partial
import PIL.Image
from naive import display_layers, animationfig, deprocess_image, show_save
from multiscale import tffunc, show_save, lap_normalize


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



def T(layer):
    '''Helper for getting layer output tensor'''
    return graph.get_tensor_by_name("import/%s:0"%layer)

def resize(img, size):
    img = tf.expand_dims(img, 0)
    return tf.image.resize_bilinear(img, size)[0,:,:,:]
resize = tffunc(np.float32, np.int32)(resize)

def calc_grad_tiled(img, t_grad, t_score, tile_size=512):
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

def render_deepdream(t_obj, img0,
                     iter_n=30, step=1.5, octave_n=4, octave_scale=1.4):
    t_score = tf.reduce_mean(t_obj)  
    t_grad = tf.gradients(t_score, t_input)[0] 
    lap_norm_func = tffunc(np.float32)(partial(lap_normalize, scale_n=4))
    # split the image into a number of octaves
    img = img0
    octaves = []
    for i in range(octave_n-1):
        hw = img.shape[:2]
        lo = resize(img, np.int32(np.float32(hw)/octave_scale))
        hi = img-resize(lo, hw)
        img = lo
        octaves.append(hi)
    
    # generate details octave by octave
    for octave in range(octave_n):
        if octave>0:
            hi = octaves[-octave]
            img = resize(img, hi.shape[:2])+hi
        for i in range(iter_n):
            g,score = calc_grad_tiled(img, t_grad,t_score)
            # g = lap_norm_func(g)
            img += g*(step / (np.abs(g).mean()+1e-7))
            print('Step ',i," Score:", score)
    show_save(img,_save, _gamma, _save_name)


img0 = PIL.Image.open(os.path.expanduser("~/Downloads/gibbon.jpg"))
img0 = np.float32(img0)
obj = T(arg[unit][0])
render_deepdream(obj,img0,octave_n=3)