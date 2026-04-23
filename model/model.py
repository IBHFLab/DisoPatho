import tensorflow as tf
from Encoder import Encoder
from GateDXHCEncoder import GateCrossEncoder
# from GateRDXHCEncoder import GateCrossEncoder


mid_site = 110
lene_site= 222
lenp_site = lene_site - 1

def binary_focal_loss(gamma=2, alpha=0.25):
    alpha = tf.constant(alpha, dtype=tf.float32)
    gamma = tf.constant(gamma, dtype=tf.float32)

    def binary_focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        alpha_t = y_true * alpha + (tf.ones_like(y_true) - y_true) * (1 - alpha)

        p_t = y_true * y_pred + (tf.ones_like(y_true) - y_true) * (
                tf.ones_like(y_true) - y_pred) + tf.keras.backend.epsilon()
        focal_loss = - alpha_t * tf.pow((tf.ones_like(y_true) - p_t), gamma) * tf.math.log(p_t)
        return tf.reduce_mean(focal_loss)

    return binary_focal_loss_fixed

def gxDencoder(input,input2):
    sequence = tf.keras.layers.Dense(1024)(input)
    sequence = tf.keras.layers.Dense(512)(sequence)
    sequence = tf.keras.layers.Dense(256)(sequence)
    
    sequence2 = tf.keras.layers.Dense(512)(input2)
    sequence2 = tf.keras.layers.Dense(256)(sequence2)
    seqoutput = GateCrossEncoder(2, 256, 8, 1024, 2, rate=0.3)(sequence,sequence2)
    seqoutput = seqoutput[:, mid_site, :]
    return seqoutput 

def processsite(input):
    sequence = tf.keras.layers.Dense(512)(input)
    sequence = tf.keras.layers.Dense(256)(sequence)
    sequence = sequence[:, mid_site, :]
    return sequence

def tanhexp(inputs):

    return inputs * tf.math.tanh(tf.math.exp(inputs))

def get_model():
    
    inputPGLM = tf.keras.layers.Input(shape=(lenp_site, 2560))
    inputESM = tf.keras.layers.Input(shape=(lene_site, 1152))
    inputAIUENG = tf.keras.layers.Input(shape=(lenp_site))

    cls_esm=inputESM[:,0,:]

    cls_feature = tf.keras.layers.Dense(512)(cls_esm)
    cls_feature = tf.keras.layers.Dense(256)(cls_feature)
    
    eng_feature = inputAIUENG[:, mid_site:mid_site+1]
    eng_feature = tf.keras.layers.Dense(32)(eng_feature)

    sequence_esm=inputESM[:,1:,:]

    site_esm = processsite(sequence_esm)

    site_pglm = gxDencoder(inputPGLM, sequence_esm)

    sequenceconcat = tf.keras.layers.Concatenate()([site_pglm, site_esm, cls_feature, eng_feature])
    feature = tf.keras.layers.Dense(800, activation=tanhexp)(sequenceconcat)
    feature = tf.keras.layers.Dropout(0.4)(feature)
    feature = tf.keras.layers.Dense(512, activation=tanhexp)(feature)
    feature = tf.keras.layers.Dropout(0.4)(feature)
    feature = tf.keras.layers.Dense(256, activation=tanhexp)(feature)
    feature = tf.keras.layers.Dropout(0.4)(feature)
    feature = tf.keras.layers.Dense(128, activation=tanhexp)(feature)
    feature = tf.keras.layers.Dropout(0.4)(feature)
    y = tf.keras.layers.Dense(1, activation='sigmoid')(feature)

    qa_model = tf.keras.models.Model(inputs=[inputPGLM, inputESM, inputAIUENG], outputs=y)
    adam = tf.keras.optimizers.Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipnorm=1.0, clipvalue=0.5)
    qa_model.compile(loss=[binary_focal_loss(alpha=.32, gamma=2)], optimizer=adam, metrics=['accuracy'])
    # qa_model.compile(loss=[binary_focal_loss(alpha=.23, gamma=2)], optimizer=adam, metrics=['accuracy'])
    
    qa_model.summary()
    return qa_model
