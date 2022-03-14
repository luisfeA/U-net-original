from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model
import tensorflow as tf

def conv_block(inputs, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

def encoder_block(inputs, num_filters):
    s = conv_block(inputs, num_filters)
    p = MaxPool2D((2, 2))(s)
    return s, p

def decoder_block(inputs, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(inputs)

    skip_features_size = skip_features.shape[1]
    inputs_size = x.shape[1]

    delta = skip_features_size - inputs_size
    delta = delta // 2

    dif = (skip_features_size+delta) - (inputs_size+delta)

    skip_features_size2 = skip_features.shape[2]
    inputs_size2 = x.shape[2]

    delta2 = skip_features_size2 - inputs_size2
    delta2 // 2

    dif2 = (skip_features_size2) - (inputs_size2-delta2)
    print(delta2)
    print(skip_features.shape)
    print(x.shape)
    skip_features = skip_features[:,delta:(skip_features_size+delta)-dif,:skip_features_size2-delta2,:]
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

def build_unet(input_shape):
    """ Input layer """
    inputs = Input(input_shape)

    """ Encoder """
    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    #s4, p4 = encoder_block(p3, 512)

    """ Bottleneck """
    b1 = conv_block(p1, 1024)

    """ Decoder """
    #d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(b1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    """ Output layer """
    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)
    #rank_4_tensor = tf.zeros([768,576,1])
    #x = tf.add(outputs, rank_4_tensor)
    outputs = tf.image.resize(outputs, [580, 780])
    #print(x.shape)
    model = Model(inputs, outputs, name="UNET")
    return model

if __name__ == "__main__":
    model = build_unet((580, 780, 3))
    model.summary()