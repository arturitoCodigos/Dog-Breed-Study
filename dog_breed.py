# Importando bibliotecas padrão
import tensorflow_datasets as tfds
import tensorflow as tf

(train, test) = tfds.load("stanford_dogs", split=["train", "test"], as_supervised=True, batch_size=32) # Carregando o dataset


# Pré-processamento das imagens
def to_scale(image, label):
    image = tf.cast(image, tf.float32)
    image = image / 255.0
    image = tf.image.resize(image, (224, 224))
    return image, label


train = train.map(lambda x, y: (x, tf.one_hot(y, depth=120)))  # One Hot encode das labels
test = test.map(lambda x, y: (x, tf.one_hot(y, depth=120)))  # One Hot encode das labels
train = train.map(to_scale)
test = test.map(to_scale)

# Configuração da rede
neuralNet = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(256, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(512, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(120, activation="softmax")
])

# Encontrei esta loss na documentação do Keras. Ela serve para multiclass.
neuralNet.compile(loss="categorical_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])

neuralNet.fit(train, epochs=10, batch_size=32)
