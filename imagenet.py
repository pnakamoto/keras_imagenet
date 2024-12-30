import keras
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

# Carrega o modelo ResNet50 pré-treinado com os pesos do ImageNet
model = ResNet50(weights='imagenet')

# Caminho da imagem a ser classificada
img_path = '.jpg'  # Substitua pelo caminho da sua imagem

# Carrega e pré-processa a imagem
img = keras.utils.load_img(img_path, target_size=(224, 224))  # Redimensiona a imagem para 224x224
x = keras.utils.img_to_array(img)  # Converte a imagem em um array NumPy
x = np.expand_dims(x, axis=0)  # Adiciona uma dimensão extra para compatibilidade com o modelo
x = preprocess_input(x)  # Normaliza os valores da imagem para o formato esperado pelo modelo

# Faz a predição
preds = model.predict(x)

# Decodifica os resultados em uma lista de tuplas (classe, descrição, probabilidade)
# (uma lista para cada amostra no batch)
print('Predicted:', decode_predictions(preds, top=3)[0])  # Mostra as 3 classes mais prováveis
