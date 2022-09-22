# Importando bibliotecas padrão
import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

(train, test) = tfds.load("stanford_dogs", split=["train", "test"], as_supervised=True) # Carregando o dataset

# Pré-processamento das imagens
def to_scale(image, label):
  image = tf.image.resize(image, (224, 224))
  label = tf.one_hot(label, 120)
  return image, label

train = train.map(to_scale)
test = test.map(to_scale)

train = train.batch(batch_size=64, drop_remainder=True)
train = train.prefetch(tf.data.AUTOTUNE)

test = test.batch(batch_size=64, drop_remainder=True)

# Camada para augmentação de imagem
img_augmentation = Sequential(
    [
        layers.RandomRotation(factor=0.15),
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        layers.RandomFlip(),
        layers.RandomContrast(factor=0.1),
    ],
    name="img_augmentation",
)

# Modelo 1 --> Soprado da documentação do keras

inputs = tf.keras.layers.Input(shape=(224,224,3))
pre_treinada_saida = img_augmentation(inputs)
neuralNet = tf.keras.applications.EfficientNetB0(weights="imagenet", input_tensor=pre_treinada_saida, include_top=False)

neuralNet.trainable = False # O modelo base nao treina

pre_treinada_saida = tf.keras.layers.GlobalAveragePooling2D()(neuralNet.output)
pre_treinada_saida = tf.keras.layers.BatchNormalization()(pre_treinada_saida)
pre_treinada_saida = tf.keras.layers.Dropout(0.2)(pre_treinada_saida)
out = tf.keras.layers.Dense(120, activation="softmax")(pre_treinada_saida)  # Output

# Construção do modelo
neuralNet = tf.keras.Model(inputs, out)

#neuralNet.summary() # Resumo da rede
    
# Learning rate decay

neuralNet.compile(loss="categorical_crossentropy",
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=["accuracy"])

neuralNet.fit(train, epochs=70, validation_data=test, verbose=1)

test_result = neuralNet.evaluate(test)
test_result = dict(zip(neuralNet.metrics_names, test_result))
print(test_result)

neuralNet.save("./savedModels_version5/")
