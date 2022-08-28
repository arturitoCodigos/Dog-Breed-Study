import tensorflow as tf
import tensorflow_datasets as tfds


def norm_input(image, label):
    return tf.cast(image, tf.float32)/255.0, label  # Normalizar entre 0 e 1. obs.: necessário converter p/ float


# Normalizando os dados
(train, test) = tfds.load('horses_or_humans',
                          split=["train", "test"],
                          as_supervised=True)
train = train.map(norm_input)
test = test.map(norm_input)

# Manipulando dados para funcionar como entrada.
# Na primeira vez que tentei treinar o modelo tive alguns erros expostos no console.
# Fiz algumas rápidas pesquisas desses error logs e encontrei soluções, mas não sei ao certo
# o que cada método aqui faz
train = train.batch(32)
test = test.batch(32)

# Montando a rede neural
neuralNet = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", kernel_regularizer="l2"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", kernel_regularizer="l2"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(16, (3, 3), activation="relu", kernel_regularizer="l2"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
])

# Hiperparâmetros da rede.
# OBS: ainda não vi as aulas sobre otimizadores e ainda não conheci melhor essa loss.
# Estou as utilizando após pesquisar na documentação e após testes empíricos com outras.
neuralNet.compile(
    loss="binary_crossentropy",
    optimizer=tf.keras.optimizers.Adam(),
    metrics=["accuracy"]
)

neuralNet.fit(train, epochs=10)

test_acc = neuralNet.evaluate(test, verbose=1)[1]

print("Results: ", test_acc)






