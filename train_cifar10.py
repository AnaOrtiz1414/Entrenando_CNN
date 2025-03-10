import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Cargar CIFAR-10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalización

# Etiquetas de la data
clases = ['Avión', 'Automóvil', 'Pájaro', 'Gato', 'Ciervo', 
          'Perro', 'Rana', 'Caballo', 'Barco', 'Camión']

# Aquí se muestran algunas imágenes. 
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(x_train[i])
    ax.set_title(clases[y_train[i][0]])
    ax.axis("off")
plt.show()

# Modelo CNN (LeNet)
model = Sequential([
    Conv2D(6, kernel_size=(5,5), activation='relu', input_shape=(32,32,3)),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(16, kernel_size=(5,5), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(120, activation='relu'),
    Dense(84, activation='relu'),
    Dense(10, activation='softmax')
])

# Compilar modelo
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Entrenar modelo con historial
history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# Guardar modelo
model.save("cifar10_model.h5")
print("Modelo guardado como cifar10_model.h5")

# Grafica da pérdida y precisión
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Pérdida
ax1.plot(history.history['loss'], label='Pérdida Entrenamiento')
ax1.plot(history.history['val_loss'], label='Pérdida Validación')
ax1.set_title('Pérdida durante el entrenamiento')
ax1.set_xlabel('Épocas')
ax1.set_ylabel('Pérdida')
ax1.legend()

# Precisión
ax2.plot(history.history['accuracy'], label='Precisión Entrenamiento')
ax2.plot(history.history['val_accuracy'], label='Precisión Validación')
ax2.set_title('Precisión durante el entrenamiento')
ax2.set_xlabel('Épocas')
ax2.set_ylabel('Precisión')
ax2.legend()

plt.show()