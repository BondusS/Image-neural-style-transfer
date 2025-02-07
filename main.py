import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import PIL.Image
import tqdm


# Convertation from tensor form to standart image form
def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)


# Loading, convert to tensor, reshape of image
def load_img(path_to_img):
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim
    new_shape = tf.cast(shape * scale, tf.int32)
    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img


# Image visualisation using matplotlib
def imshow(image, title=None):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)
    plt.imshow(image)
    if title:
        plt.title(title)


# Returns layers of architecture VGG19
def mobnet_layers(layer_names):
    mobnet = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    mobnet.trainable = False
    outputs = [mobnet.get_layer(name).output for name in layer_names]
    model = tf.keras.Model([mobnet.input], outputs)
    return model


# Computes Gram matrix which shows current condition of tensor under work
def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result/num_locations


# Scales values to [0.0, 1.0]
def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


# Computes current status by MSE loss of content and style tensors
def style_content_loss(outputs):
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2)
                           for name in style_outputs.keys()])
    style_loss *= style_weight / len(style_outputs)  # num_style_layers
    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2)
                             for name in content_outputs.keys()])
    content_loss *= content_weight / len(content_outputs)  # num_content_layers
    loss = style_loss + content_loss
    return loss


# Makes step of training by step of gradient optimiser and regularisation by weighted total variations of image
@tf.function()
def train_step(image):
    with tf.GradientTape() as tape:
        outputs = extractor(image)
        loss = style_content_loss(outputs)
        loss += 25 * tf.image.total_variation(image)
    grad = tape.gradient(loss, image)
    opt.apply_gradients([(grad, image)])
    image.assign(clip_0_1(image))


# Describes class of set of objects under work
class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.mobnet = mobnet_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.mobnet.trainable = False

    def call(self, inputs):
        inputs = inputs*255.0
        preprocessed_input = tf.image.resize(inputs, (224, 224))
        # preprocessed_input = tf.keras.applications.VGG19.preprocess_input(inputs)
        outputs = self.mobnet(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                          outputs[self.num_style_layers:])
        style_outputs = [gram_matrix(style_output)
                         for style_output in style_outputs]
        content_dict = {content_name: value
                        for content_name, value
                        in zip(self.content_layers, content_outputs)}
        style_dict = {style_name: value
                      for style_name, value
                      in zip(self.style_layers, style_outputs)}
        return {'content': content_dict, 'style': style_dict}


# Loading of images under work
content_path = 'MSU.jpg'
style_path = 'minions.jpg'
content_image = load_img(content_path)
style_image = load_img(style_path)

# Images visualisation using matplotlib
plt.subplot(1, 2, 1)
imshow(content_image, 'Content Image')
plt.subplot(1, 2, 2)
imshow(style_image, 'Style Image')
plt.show()

# Loading architecture VGG19 with weights trained on imagenet
mobnet = tf.keras.applications.VGG19(include_top=False, weights='imagenet')

# Shows all layers of architecture VGG19
"""
print()
for layer in mobnet.layers:
    print(layer.name)
"""

# Sets layers which uses in transformation
content_layers = ['block5_conv2']
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']

# Shows number of layers in each set
"""
num_content_layers = len(content_layers)
num_style_layers = len(style_layers)
print(num_content_layers, num_style_layers)
"""

# Initialise model and target condition
style_extractor = mobnet_layers(style_layers)
style_outputs = style_extractor(style_image*255)

# Shows architecture of initialised model and stats of layers output neurons
"""
for name, output in zip(style_layers, style_outputs):
    print(name)
    print("  shape: ", output.numpy().shape)
    print("  min: ", output.numpy().min())
    print("  max: ", output.numpy().max())
    print("  mean: ", output.numpy().mean())
    print()
"""

# Initialise extractor and start condition
extractor = StyleContentModel(style_layers, content_layers)
results = extractor(tf.constant(content_image))

# Shows architectures of initialised models
"""
print('Styles:')
for name, output in sorted(results['style'].items()):
    print("  ", name)
    print("    shape: ", output.numpy().shape)
    print("    min: ", output.numpy().min())
    print("    max: ", output.numpy().max())
    print("    mean: ", output.numpy().mean())
    print()
print("Contents:")
for name, output in sorted(results['content'].items()):
    print("  ", name)
    print("    shape: ", output.numpy().shape)
    print("    min: ", output.numpy().min())
    print("    max: ", output.numpy().max())
    print("    mean: ", output.numpy().mean())
"""

# Initialise extractors, optimiser, start values of variables
style_targets = extractor(style_image)['style']
content_targets = extractor(content_image)['content']
image = tf.Variable(content_image)
opt = tf.keras.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
style_weight = 1e-2
content_weight = 1e4

# Launch process of models work with monitoring
for i in tqdm.tqdm(range(100)):
    train_step(image)
plt.imshow(tensor_to_image(image))
plt.show()
