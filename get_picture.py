from keras.layers import Input, Lambda, Dense, Flatten
from keras.layers import AveragePooling2D, MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.models import Model, Sequential
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import keras.backend as K

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import fmin_l_bfgs_b
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

img_style = 'art/edvard_munch.jpg'
img_content = 'images/cat.jpg'


def unpreprocess(img):
  img[..., 0] += 103.939
  img[..., 1] += 116.779
  img[..., 2] += 126.68
  img = img[..., ::-1]
  img = np.clip(img, 0, 255).astype('uint8')
  return img

def get_images(path_style, path_content, shape):
    img_style = load_img(path_style, target_size = shape)
    img_content = load_img(path_content, target_size=shape)
    img_style = img_to_array(img_style)
    img_content = img_to_array(img_content)

    img_style = np.expand_dims(img_style, axis = 0)
    img_content = np.expand_dims(img_content, axis = 0)

    img_style = preprocess_input(img_style)
    img_content = preprocess_input(img_content)

    return img_style, img_content

def create_content_model(model, layer):
    new_model = Model(inputs = model.input, outputs = model.layers[layer].output)
    return new_model

def create_style_model(model):
    s_layers_output = [layer.get_output_at(1) for layer in model.layers if layer.name.endswith('conv1')]
    s_model = Model(inputs = model.input, outputs = s_layers_output)

    return s_model, s_layers_output


def gram_matrix(x):
    X = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(X, K.transpose(X)) / x.get_shape().num_elements()
    return gram


def content_loss(F, model):
    c_loss = K.mean(K.square(F-model.output))
    c_grad = K.gradients(c_loss, model.input)
    return c_loss, c_grad

def style_loss(model, target_output, style_output):
    w = 0.25 # from acticle arXiv:1508.06576v2
    loss = 0
    for t, s in zip(target_output, style_output):
        loss += w * K.mean(K.square(gram_matrix(t[0]) - gram_matrix(s[0])))
    grad = K.gradients(loss, model.input)

    return loss, grad


def VGG16_AvgPool(shape):
  # we want to account for features across the entire image
  # so get rid of the maxpool which throws away information
  vgg = VGG16(input_shape=shape, weights='imagenet', include_top=False)

  new_model = Sequential()
  for layer in vgg.layers:
    if layer.__class__ == MaxPooling2D:
      # replace it with average pooling
      new_model.add(AveragePooling2D())
    else:
      new_model.add(layer)

  return new_model

def VGG16_AvgPool_CutOff(shape, num_convs):

  model = VGG16_AvgPool(shape)
  new_model = Sequential()
  n = 0
  for layer in model.layers:
    if layer.__class__ == Conv2D:
      n += 1
    new_model.add(layer)
    if n >= num_convs:
      break

  return new_model

shape = (306,512,3)


img_style, img_content = get_images(img_style, img_content, shape)
batch_shape = img_content.shape
shape = img_content.shape[1:]

model = VGG16(weights='imagenet', include_top = False, input_shape = shape)



vgg_avg = VGG16_AvgPool(shape)
content_model = Model(vgg.input, vgg.layers[13].get_output_at(0))
# c_model = create_content_model(model, 15)
s_model, s_layers = create_style_model(vgg_avg)

F = K.variable(content_model.predict(img_content))

A = [K.variable(target) for target in s_model.predict(img_style)]


c_loss, c_grad = content_loss(F, content_model)
s_loss, s_grad = style_loss(s_model, A, s_layers)


total_loss = c_loss + s_loss

grad = K.gradients(total_loss, vgg_avg.input)

# get_loss_grads_c = K.function(inputs = [c_model.input], outputs = [c_loss] + c_grad)
# get_loss_grads_s = K.function(inputs = [s_model.input], outputs = [s_loss] + s_grad)

get_loss_grads = K.function(inputs = [vgg_avg.input], outputs = [total_loss] + grad)

def get_loss_and_grads_wrapper(x_vec):
    #Transform shape of image to vector for scipy function
    #input must be float64
    l, g = get_loss_grads([x_vec.reshape(*batch_shape)])
    return l.astype(np.float64), g.flatten().astype(np.float64)


x = np.random.randn(np.prod(batch_shape))
losses = []
for i in range(10):
    x, l, _ = fmin_l_bfgs_b(
      func=get_loss_and_grads_wrapper,
      x0=x,

      maxfun=20
    )
    x = np.clip(x, -127, 127)

    print("iter=%s, loss=%s" % (i, l))
    losses.append(l)

img = x.reshape(*batch_shape)
img=unpreprocess(img)
plt.imshow(img[0])
plt.axis('off')
plt.savefig('img_generated/cat_munch.png')
plt.clf()

#
# plt.imshow(img[0])
# plt.show()
