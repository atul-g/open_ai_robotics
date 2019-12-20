import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential



def create_actor_network():
    model=tf.keras.Sequential([
                           tf.keras.layers.Dense(256, input_shape=(28,), activation='relu'),
                           tf.keras.layers.Dense(256, activation='relu'),
                           tf.keras.layers.Dense(256, activation="relu"),
                           tf.keras.layers.Dense(4, activation='tanh') ##### 4 possible actions
    ])

    return model
    
    
def create_critic_network():
    model=tf.keras.Sequential([
                           tf.keras.layers.Dense(256, input_shape=(31,), activation='relu'),
                           tf.keras.layers.Dense(256, activation='relu'),
                           tf.keras.layers.Dense(256, activation="relu"),
                           tf.keras.layers.Dense(1, activation='linear') ##### 1 q-value for an action and state
    ])
    model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
    return model
    
a=create_actor_network()
o=a()
print(a.summary())


