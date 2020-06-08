import tensorflow as tf
from tensorflow import keras

def Chara_CNN_BGRU(vocab_size, seq_length, embed_size,
                   num_filters=128, kernel_sizes='2,3,5',
                   gru_units=256, dropout_rate=0.5, regularizers_lambda=0.0001,
                   num_classes=23):
    # [batch_size, seq_length]
    inputs = keras.Input(shape=(seq_length,), name='input_data')  # shape 这个参数不需要包括 batch_size
    
    # [batch_size, seq_length, embed_size]
    embed_initer = keras.initializers.RandomUniform(minval=-1, maxval=1)
    embed = tf.reshape(
        keras.layers.Embedding(
            vocab_size, embed_size, embeddings_initializer=embed_initer,
            name='embedding_layer')(inputs),
        (-1, seq_length, embed_size, 1)
        )
    # [batch_size, seq_length, embed_size, 1]
    
    # TextCNN
    # ----------------------------------------------------------
    pools = []
    convs = []
    for kernel_size in map(int, kernel_sizes.replace(' ', '').split(',')):
        # [batch_size, seq_length, 1, num_filters]
        filter_shape = [kernel_size, embed_size]
        '''
        # data_format: default --> channel_last
        #   channel_first: batch，in_channels，in_height，in_width
        #   channel_last: batch, in_height, in_width, in_channels
        '''
        pad_length = kernel_size-1
        left_seq_padding_size = pad_length // 2 # 不为偶数时，右比左多pad一个，下比上多pad一个
        right_seq_padding_size = pad_length - left_seq_padding_size
        padded_embed = keras.layers.ZeroPadding2D(
            padding=((left_seq_padding_size, right_seq_padding_size), (0,0)),
            data_format='channels_last')(embed)
        conv = keras.layers.Conv2D(num_filters, filter_shape,
                                   strides=(1,1), padding='valid',
                                   data_format='channels_last', activation='relu',
                                   kernel_initializer='glorot_normal',
                                   bias_initializer=keras.initializers.constant(0.1),
                                   name='convolution_{:d}'.format(kernel_size))(padded_embed)
        # [batch_size, 1, 1, num_filters]
        max_pool_shape = (seq_length ,1)
        max_pool = keras.layers.MaxPool2D(
            pool_size=max_pool_shape, strides=(1,1), padding='valid',
            data_format='channels_last', 
            name='max_pooling_{:d}'.format(kernel_size))(conv)
        
        convs.append(conv)
        pools.append(max_pool)
    # [batch_size, seq_length, 1, num_filters * f]
    pool_outputs = keras.layers.concatenate(pools, axis=-1, name='concatenate_pool')
    # [batch_size, ...]
    cnn_output = keras.layers.Flatten(data_format='channels_last', name='cnn_flatten')(pool_outputs)
    
    
    # Bi-GRU
    # -------------------------------------------------------- [batch_size, seq_length, 1, num_filters]
    # [batch_size, seq_length, 1, num_filters * k]
    conv_outputs = tf.squeeze(
        keras.layers.concatenate(convs, axis=-1, name='concatenate_conv'),
        axis=2)
    gru_outputs = keras.layers.Bidirectional(
        keras.layers.GRU(units=gru_units, return_sequences=False, return_state=True), name='bi-gru')(conv_outputs)
    
    # Concate
    outputs = keras.layers.concatenate([gru_outputs[0], cnn_output], axis=-1, name='concatenate_output')
    outputs = keras.layers.Dropout(rate=dropout_rate, name='dropout')(outputs)
    
    outputs = keras.layers.Dense(num_classes, activation='softmax',
                                 kernel_initializer='glorot_normal',
                                 bias_initializer=keras.initializers.constant(0.1),
                                 kernel_regularizer=keras.regularizers.l2(regularizers_lambda),
                                 name='dense')(outputs)
    
    # Model
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model
