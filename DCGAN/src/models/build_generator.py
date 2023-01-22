from tensorflow.keras import layers
from tensorflow.keras.models import Model

def build_generator(inputs, image_size)->Model:
    """Build a generator network model
    
    Stack of BN-ReLU-Conv2DTranspose to  generate fake images
    Output activation is sigmoid instead of tanh
    Sigmoid converges easily

    Arguments:
        inputs {Layer}: Input layer of the generator, the z-vector
        image_size {tensor}: Target size of one side {assuming square image}

    Returns:
        generator {Model}: Generator Model
    """

    image_resize = image_size // 4
    # network parameters
    kernel_size = 5
    layers_filters = [128, 64, 32, 1]

    x = layers.Dense(image_resize * image_resize * layers_filters[0])(inputs)
    x = layers.Reshape((image_resize, image_resize, layers_filters[0]))(x)

    for filters in layers_filters:
        # first two convolutional layers use strides = 2
        # the last two use strides = 1
        if filters > layers_filters[-2]: 
            strides = 2
        else: 
            strides = 1
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2DTranspose(filters=filters,
                                   kernel_size=kernel_size,
                                   strides=strides,
                                   padding='same')(x)

    x = layers.Activation('sigmoid')(x)
    generator = Model(inputs, x, name='generator')
    return generator
