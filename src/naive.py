import os
import numpy as np
from scipy.misc import imsave, toimage
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tensorflow as tf
import cv2
import PIL.Image

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

layers = [op.name for op in graph.get_operations() if 'import/' in op.name]


# Show all the layers in this model
def display_layers():
    print('-- Number of layers can use', len(layers))
    for i in layers:
        print(i[7:])


def T(layer):
    '''Helper for getting layer output tensor'''
    return graph.get_tensor_by_name("import/%s:0"%layer)



def animationfig(imagelist,interval=500):
    fig = plt.figure()
    im = plt.imshow(imagelist[0])
    def updatefig(j):
        # set the data in the axesimage object
        im.set_array(imagelist[j])
        return [im]
    ani = animation.FuncAnimation(fig, updatefig, frames=range(len(imagelist)), 
                    interval=interval, repeat_delay= 2000, blit=True)

    plt.show()

def gammatrick(img,log=3,bias = 0.23):
    img += bias
    img[img>=1.0]=0.88
    return (img)**log

def deprocess_image(x,gamma=False):
    # don't change x
    x1 = x.copy()
    # normalize tensor: center on 0., ensure std is 0.1
    x1 -= x1.mean()
    '''Normalize the image range for visualization'''
    x1 /= (x1.std() + 1e-7)
    x1 *= 0.1

    # clip to [0, 1]
    x1 += 0.5
    x1 = np.clip(x1, 0, 1)
    if gamma==True:
        x1 =gammatrick(x1)
    # convert to RGB array
    x1 *= 255
    x1 = np.clip(x1, 0, 255).astype('uint8')
    return x1


def show_save(x,save=False,gamma = False,name='tensor'):
    temp = deprocess_image(x,gamma)
    toimage(temp).show()
    if save==True:
        img = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.expanduser("~/Desktop/"+name+".jpg"), img)



def render_naive(t_obj, img0, iter_n=20, step=1.0,name ='tensorflower1'):
    t_score = tf.reduce_mean(t_obj)  # defining the optimization objective
    t_grad = tf.gradients(t_score, t_input)[0] # behold the power of automatic differentiation!
    img = img0.copy()
    imagelist.append(deprocess_image(img))
    for i in range(iter_n):
        g, score, _shape = sess.run([t_grad, t_score, t_obj], {t_input:img})

        # normalizing the gradient, so the same step size should work 
        g /= g.std() + 1e-8         # for different layers and networks
        img += g*step   #- img*0.1 #l2
        print("Step ",i,", Loss value:",score)
        imagelist.append(deprocess_image(img))

    show_save(img,name=name)

    return img

def interpolate_layer(fn,l1,l2, c1, c2, step = 5):
    interpolate = []
    for i in np.linspace(0,1.0,num=step):
        layers = T(l1)[:,:,:,c1]*(1-i) + T(l2)[:,:,:,c2]*i
        img_noise = np.random.uniform(size=(224,224,3)) + 100.0
        _img = fn(layers,img_noise)
        interpolate.append(deprocess_image(_img))
    
    animationfig(interpolate,500)

if __name__ == '__main__':

    # Show all the layers in this model yourself
    # display_layers()  

    img_noise = np.random.uniform(size=(224,224,3)) + 100.
    imagelist = [] # for animation 

    render_naive(T(arg[unit][0])[:,:,:,arg[unit][1]], img_noise, iter_n=20, step=1.0,name ='img')
    animationfig(imagelist,200)
    

    # interpolate_layer(render_naive,layer,layer,20,30,5)



    