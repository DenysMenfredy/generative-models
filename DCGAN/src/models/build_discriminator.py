from tensorflow.keras import layers
from tensorflow.keras.models import Model


def build_discriminator(inputs):
    """Build a Discriminator Model
    Stack of LeakyReLU-Conv2D to discriminate real from fake.
    The network does not converge with BN so it is not used here
    unlike in [1] or original paper.
    Arguments:
        inputs (Layer): Input layer of the discriminator (the image)
    Returns:
        discriminator (Model): Discriminator Model
    """
    kernel_size = 5
    layer_filters = [32, 64, 128, 256]

    x = inputs
    for filters in layer_filters:
        # first 3 convolution layers use strides = 2
        # last one uses strides = 1
        if filters == layer_filters[-1]:
            strides = 1
        else:
            strides = 2
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   strides=strides,
                   padding='same')(x)

    x = layers.Flatten()(x)
    x = layers.Dense(1)(x)
    x = layers.Activation('sigmoid')(x)
    discriminator = Model(inputs, x, name='discriminator')
    return discriminator