# Importando bibliotecas padrão
import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf

(train, test) = tfds.load("stanford_dogs", split=["train", "test"], as_supervised=True, batch_size=32) # Carregando o dataset


# Pré-processamento das imagens
def to_scale(image, label):
    image = tf.cast(image, tf.float32)
    image = image / 255.0
    image = tf.image.resize(image, (150, 150))
    return image, label


# train = train.map(lambda x, y: (x, tf.one_hot(y, depth=120))) # One Hot encode das labels
# test = test.map(lambda x, y: (x, tf.one_hot(y, depth=120))) # One Hot encode das labels
train = train.map(to_scale)
test = test.map(to_scale)

# Pré treinado como modelo base
pre_treinada = tf.keras.applications.inception_v3.InceptionV3(weights="imagenet", include_top=False)

pre_treinada_saida = pre_treinada.output
pre_treinada_saida = tf.keras.layers.GlobalAveragePooling2D()(pre_treinada_saida)
pre_treinada_saida = tf.keras.layers.Dense(256, activation="relu")(pre_treinada_saida)     # Camada interna para processar features
pre_treinada_saida = tf.keras.layers.Dense(512, activation="relu")(pre_treinada_saida)     # Camada interna para processar features
pre_treinada_saida = tf.keras.layers.Dense(120, activation="softmax")(pre_treinada_saida)  # Output

# Construção do modelo
neuralNet = tf.keras.Model(inputs=pre_treinada.input, outputs=pre_treinada_saida)

for camada in pre_treinada.layers:
    camada.trainable = False # Quero treinar apenas a minha parte

neuralNet.compile(loss="sparse_categorical_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])

neuralNet.fit(train, epochs=6, batch_size=32)

test_result = neuralNet.evaluate(test)
test_result = dict(zip(neuralNet.metrics_names, test_result))
print(test_result)

neuralNet.save("./savedModels/")
