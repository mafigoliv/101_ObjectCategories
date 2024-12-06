import os
import random
import numpy as np
import keras
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, MaxPooling2D

# Função auxiliar para carregar a imagem e pré-processar
def get_image(path):
    img = image.load_img(path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return img, x

# Caminho da pasta de imagens
root = '.'
train_split, val_split = 0.7, 0.15

categories = ['Cat', 'Dog']

data = []
for c, category in enumerate(categories):
    category_path = os.path.join(root, category)
    print(f"Processando categoria: {category_path}")
    images = [os.path.join(dp, f) for dp, dn, filenames 
              in os.walk(category_path) for f in filenames 
              if os.path.splitext(f)[1].lower() in ['.jpg', '.png', '.jpeg']]
    print(f"Encontradas {len(images)} imagens na categoria {category}")
    for img_path in images:
        _, x = get_image(img_path)
        data.append({'x': np.array(x[0]), 'y': c})

# Verificar a existência de dados
if not data:
    raise ValueError("Nenhuma imagem foi carregada. Verifique os caminhos e extensões de arquivo.")

num_classes = len(categories)
random.shuffle(data)

# Verificar a divisão dos dados
def verificar_divisao_dados(data, train_split, val_split):
    idx_val = int(train_split * len(data))
    idx_test = int((train_split + val_split) * len(data))
    train = data[:idx_val]
    val = data[idx_val:idx_test]
    test = data[idx_test:]

    print(f"Tamanho do conjunto de treino: {len(train)}")
    print(f"Tamanho do conjunto de validação: {len(val)}")
    print(f"Tamanho do conjunto de teste: {len(test)}")

    if len(train) == 0 or len(val) == 0 or len(test) == 0:
        raise ValueError("Uma das divisões (treino, validação ou teste) está vazia.")
    
    return train, val, test

# Verificar se as divisões não estão vazias
train, val, test = verificar_divisao_dados(data, train_split, val_split)

x_train = np.array([t["x"] for t in train])
y_train = np.array([t["y"] for t in train])
x_val = np.array([t["x"] for t in val])
y_val = np.array([t["y"] for t in val])
x_test = np.array([t["x"] for t in test])
y_test = np.array([t["y"] for t in test])

# Verificar novamente se as matrizes não estão vazias
print(f"Training data: {x_train.shape}, {y_train.shape}")
print(f"Validation data: {x_val.shape}, {y_val.shape}")
print(f"Testing data: {x_test.shape}, {y_test.shape}")

# Normalizar os dados
x_train = x_train.astype('float32') / 255.
x_val = x_val.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# Converter rótulos para vetores one-hot
y_train = keras.utils.to_categorical(y_train, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Resumo das informações carregadas
print(f"Terminou de carregar {len(data)} imagens de {num_classes} categorias")
print(f"Divisão de treino / validação / teste: {len(x_train)}, {len(x_val)}, {len(x_test)}")
print(f"Forma dos dados de treino: {x_train.shape}")
print(f"Forma dos rótulos de treino: {y_train.shape}")

# Construir a rede neural
model = Sequential([
    Conv2D(32, (3, 3), input_shape=x_train.shape[1:]),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(32, (3, 3)),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Dropout(0.25),

    Conv2D(32, (3, 3)),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(32, (3, 3)),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Dropout(0.25),

    Flatten(),
    Dense(256),
    Activation('relu'),

    Dropout(0.5),

    Dense(num_classes),
    Activation('softmax')
])

model.summary()

# Compilar o modelo
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Treinar o modelo
history = model.fit(x_train, y_train, batch_size=128, epochs=10, validation_data=(x_val, y_val))

# Plotar loss e accuracy de validação
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4))
ax1.plot(history.history["val_loss"])
ax1.set_title("Validation Loss")
ax1.set_xlabel("Epochs")

ax2.plot(history.history["val_accuracy"])
ax2.set_title("Validation Accuracy")
ax2.set_xlabel("Epochs")
ax2.set_ylim(0, 1)
plt.show()

# Avaliar o modelo com dados de teste
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', loss)
print('Test accuracy:', accuracy)

# Usar o modelo VGG16 pré-treinado
vgg = keras.applications.VGG16(weights='imagenet', include_top=True)
vgg.summary()

# Criar novo modelo com a camada de classificação personalizada
inp = vgg.input
new_classification_layer = Dense(num_classes, activation='softmax')(vgg.layers[-2].output)
model_new = Model(inp, new_classification_layer)

# Congelar todas as camadas exceto a última
for layer in model_new.layers[:-1]:
    layer.trainable = False

# Compilar o novo modelo
model_new.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_new.summary()

# Treinar o novo modelo
history2 = model_new.fit(x_train, y_train, batch_size=128, epochs=10, validation_data=(x_val, y_val))

# Plotar loss e accuracy de validação para o novo modelo
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4))
ax1.plot(history.history["val_loss"], label='Base Model')
ax1.plot(history2.history["val_loss"], label='VGG16 Model')
ax1.set_title("Validation Loss")
ax1.set_xlabel("Epochs")
ax1.legend()

ax2.plot(history.history["val_accuracy"], label='Base Model')
ax2.plot(history2.history["val_accuracy"], label='VGG16 Model')
ax2.set_title("Validation Accuracy")
ax2.set_xlabel("Epochs")
ax2.set_ylim(0, 1)
ax2.legend()
plt.show()

# Avaliar o novo modelo com dados de teste
loss, accuracy = model_new.evaluate(x_test, y_test, verbose=0)
print('Test loss:', loss)
print('Test accuracy:', accuracy)

# Predizer com uma imagem de teste
img_path = 'Cat/3.jpg'  # Substitua pelo caminho correto da imagem
img, x = get_image(img_path)
probabilities = model_new.predict(x)
print(f"Probabilidades: {probabilities}")
