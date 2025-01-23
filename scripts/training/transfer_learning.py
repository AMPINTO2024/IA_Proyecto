from keras import Model
from keras.src.callbacks import EarlyStopping, ModelCheckpoint, Callback
import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
from keras.src.layers import Flatten, Dense
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd

class BestModelEpochCallback(Callback):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.best_epoch = None

    def on_epoch_end(self, epoch, logs=None):
        # Guardar el epoch con mejor accuracy (better val_accuracy)
        if self.best_epoch is None:
            self.best_epoch = (epoch, logs['val_accuracy'])
        elif logs['val_accuracy'] > self.best_epoch[1]:
            self.best_epoch = (epoch, logs['val_accuracy'])
            print(f'Best model epoch: {self.best_epoch[0] + 1} with val_accuracy: {self.best_epoch[1]}')


def config_model_for_transfer_learning(model, num_classes):
    # Congelar las capas del modelo base
    for layer in model.layers:
        layer.trainable = False

    # Añadir capas personalizadas
    # TODO: investigar si añadir capas densas es la mejor opcion
    x = Flatten()(model.output)
    x = Dense(1024, activation='relu')(x)
    output = Dense(num_classes, activation='softmax')(x)

    # Crear un modelo nuevo a realizar transfer learning
    t_model = Model(inputs=model.input, outputs=output)
    t_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return t_model

def training_model(model, train_dir, seed, max_epochs=50, batch_size=32, patience=5,
                   path_best_model='best_model.keras'):

    # reproducibilidad
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Preparar los datos
    train_datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )
    validation_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    # Configurar callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(path_best_model, monitor='val_loss', save_best_only=True)
    best_model_epoch_callback = BestModelEpochCallback()

    # Entrenar el modelo
    history = model.fit(
        train_generator,
        epochs=max_epochs,
        validation_data=validation_generator,
        callbacks=[early_stopping, model_checkpoint, best_model_epoch_callback]
    )

    return history, best_model_epoch_callback.best_epoch[0]

# evaluate the model using test_dir, present the confusion matrix and the classification report
def evaluate_model(model, test_dir):
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=1,
        class_mode='categorical',
        shuffle=False
    )

    predictions = model.predict(test_generator)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes
    # plot results
    plot_confusion_matrix(y_true, y_pred, test_generator.class_indices)
    return y_true, y_pred


def plot_confusion_matrix(y_true, y_pred, class_indices):

    cm = confusion_matrix(y_true, y_pred)
    cm = cm / cm.sum(axis=1)[:, np.newaxis]

    df_cm = pd.DataFrame(cm, index=class_indices.keys(), columns=class_indices.keys())
    plt.figure(figsize=(10, 7))
    sns.heatmap(df_cm, annot=True, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# Plotting the learning process
def plot_learning_curves(history, best_epoch):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo-', label='Training accuracy')
    plt.plot(epochs, val_acc, 'ro-', label='Validation accuracy')
    plt.axvline(x=best_epoch + 1, color='g', linestyle='--', label='Best model selected')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo-', label='Training loss')
    plt.plot(epochs, val_loss, 'ro-', label='Validation loss')
    plt.axvline(x=best_epoch + 1, color='g', linestyle='--', label='Best model selected')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()
