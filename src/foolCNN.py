import os
import numpy as np
from scipy.misc import imsave, toimage
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tensorflow as tf
import PIL.Image
from naive import deprocess_image, display_layers
import cv2


# Class score - Saliency pixel - Fool the CNN

model_fn = '~/tensorflow_inception_graph.pb'

graph = tf.Graph()
sess = tf.InteractiveSession(graph=graph)
with tf.gfile.FastGFile(os.path.expanduser(model_fn), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())



t_input = tf.placeholder(np.float32, name='input') # define the input tensor
imagenet_mean = 117.0 # define wrt imagenet's standard
t_preprocessed = tf.expand_dims(t_input-imagenet_mean, 0)
tf.import_graph_def(graph_def, {'input':t_preprocessed})
layers = [op.name for op in graph.get_operations() if 'import/' in op.name]

def T(layer):
    '''Helper for getting layer output tensor'''
    return graph.get_tensor_by_name("import/%s:0"%layer)

def print_prob(prob, file_path='./imagenet_comp_graph_label_strings.txt'):
    synset = [l.strip() for l in open(file_path).readlines()]
  
    print("Sum prob: ", np.sum(prob))
 
    pred = np.argsort(prob)[::-1]

    top1 = (synset[pred[0]], prob[pred[0]])
    print("Top1:", top1, 'index:',np.argmax(prob))
    # Get top5 label
    top5 = [(synset[pred[i]], prob[pred[i]]) for i in range(5)]
    print("Top5: ", top5 )

    return str(top1[0] + ': ' + str(top1[1])[:5])

def fool_CNN(t_obj, img0, iter_n=3, step=1.0):
    img_noise = np.random.uniform(size=(224,224,3)) + 100.
    t_score = tf.reduce_mean(t_obj)  
    t_grad = tf.gradients(t_score, t_input)[0]  
    for i in range(iter_n):
        g, score = sess.run([t_grad, t_score], {t_input:img0})
        g /= g.std()+1e-8         
        img0 += g*step  
        img_noise +=g*step
        print("Step ",i,", Loss value:",score)
    ## Display
    score = sess.run(T('output2'),feed_dict = {t_input:img0})    
    print(" -- After fooling CNN --")
    top1 = print_prob(score[0])
    cv2.putText(img0,str(top1),(70, 220), 0 , 0.5 ,(255, 0, 0),2)
    toimage(img_noise).show()
    toimage(img0).show()
    
    

#gibbon: 185, panda: 169, bulldog: 82

img0 = PIL.Image.open(os.path.expanduser("~/Downloads/dog.jpg"))
img0 = img0.resize((224, 224), PIL.Image.NEAREST)

img0 = np.float32(img0)
img0 = np.array(img0[:,:,:3]) # crop image in OS X create a 4-dim depth image
img = img0.copy()


score = sess.run(T('output2'),feed_dict={t_input:img0})
top1 = print_prob(score[0])
cv2.putText(img,str(top1),(70, 220), 0 , 0.5 ,(255, 0, 0),2)
toimage(img).show()
fool_CNN(T("softmax2_pre_activation")[:,169],img0)

