import numpy as np
from prettytable import PrettyTable
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
from livelossplot import PlotLossesKeras
from tensorflow.keras.callbacks import ModelCheckpoint
import os

"""
    Этот скрипт реализует процесс обучения модели глубокого обучения для классификации эмоций на основе изображений. Модель использует предварительно обученную архитектуру `VGG19` в качестве базы и обучается на пользовательском наборе данных. В процессе обучения осуществляется аугментация данных, настройка гиперпараметров, сохранение чекпоинтов и отображение результатов.

    Параметры ввода:

    1. BATCH_SIZE: Размер батча, используемый в процессе обучения и валидации.
    2. IMG_SHAPE: Размер входного изображения (ширина и высота).
    3. Директория данных: Путь к папке с данными для обучения, организованными по подкаталогам (одна папка на класс).
    4. NUM_CLASSES: Количество классов (эмоций) в наборе данных.
    5. EPOCHS: Количество эпох для обучения.

"""
print('Введите размер батча:', end=' ')
BATCH_SIZE = int(input())
print('Введите размер изображения:', end=' ')
IMG_SHAPE = int(input())
print("Укажите директорию с данными для обучения:", end=' ')
n = input()
dir = Path(n)
print('Введите количество эмоций:', end=' ')
NUM_CLASSES = int(input())
print('Введите количество эпох:', end=' ')
EPOCHS = int(input())


def DataGeneration():
    """
    Создаёт объект `ImageDataGenerator` для предварительной обработки изображений.

    :return: ImageDataGenerator: объект для генерации и преобразования данных.
    :rtype: ImageDataGenerator
    """
    image_gen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                   validation_split=0.2,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode="nearest")
    return image_gen


# Создание генераторов данных для тренировки и валидации
def create_generation_for_training(BATCH_SIZE, IMG_SHAPE, dir):
    """
    Создаёт генератор данных для тренировочной выборки.

    :param: int BATCH_SIZE: Размер батча.
    :param: int IMG_SHAPE: Размер изображения (высота и ширина).
    :param: str dir: Директория с изображениями для обучения.
    :return: DirectoryIterator: Генератор данных для тренировочной выборки.
    :rtype: DirectoryIterator

    """
    train_data_gen = DataGeneration().flow_from_directory(batch_size=BATCH_SIZE,
                                                          directory=dir,
                                                          shuffle=True,
                                                          target_size=(IMG_SHAPE, IMG_SHAPE),
                                                          class_mode="categorical",
                                                          subset="training")
    return train_data_gen


def validation(BATCH_SIZE, IMG_SHAPE, dir):
    """
    Создаёт генератор данных для валидационной выборки.

    :param: int BATCH_SIZE: Размер батча.
    :param: int IMG_SHAPE: Размер изображения (высота и ширина).
    :param: str dir: Директория с изображениями для обучения.
    :return: DirectoryIterator: Генератор данных для валидационной выборки.
    :rtype: DirectoryIterator

    """
    val_data_gen = DataGeneration().flow_from_directory(batch_size=BATCH_SIZE,
                                                        directory=dir,
                                                        shuffle=False,
                                                        target_size=(IMG_SHAPE, IMG_SHAPE),
                                                        class_mode='categorical',
                                                        subset="validation")
    return val_data_gen


IMG_SHAPE1 = (IMG_SHAPE, IMG_SHAPE, 3)


def create_base_model(IMG_SHAPE1):
    """
    Создаёт базовую модель на основе VGG19 с исключением верхних слоёв (include_top=False).

    :param: tuple IMG_SHAPE1: Размер входного изображения в формате (высота, ширина, каналы).
    :return: Model: Базовая модель VGG19 без верхних слоёв.
    :rtype: keras.models.Model

    """
    base_model = tf.keras.applications.VGG19(input_shape=IMG_SHAPE1,
                                             include_top=False,
                                             weights='imagenet')
    base_model.trainable = True
    base_model.summary()
    return base_model


def create_model(NUM_CLASSES):
    """
    Создаёт полную архитектуру модели с дополнительными полносвязными слоями.

    :param: int NUM_CLASSES: Количество классов для классификации.
    :return: Sequential: Полная архитектура модели.
    :rtype: keras.models.Sequential

    """

    model = tf.keras.Sequential([create_base_model(IMG_SHAPE1),
                                 tf.keras.layers.GlobalAveragePooling2D(),
                                 tf.keras.layers.Dense(512, activation="relu"),
                                 tf.keras.layers.Dropout(0.3),
                                 tf.keras.layers.Dense(NUM_CLASSES)])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.summary()
    return model


checkpoint_path = "trainer/cp-{epoch:04d}.weights.h5"
checkpoint_dir = os.path.dirname(checkpoint_path)

latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)


# Создайте колбэк ModelCheckpoint
def create_checkpoint(checkpoint_path):
    """
    Создаёт callback для сохранения весов модели после каждой эпохи.

    :param: str checkpoint_path: Путь для сохранения весов модели.
    :return: ModelCheckpoint: Колбэк для сохранения контрольных точек.
    :rtype: ModelCheckpoint.
    """
    cp_callback = ModelCheckpoint(filepath=checkpoint_path,
                                  save_weights_only=True,
                                  verbose=1)
    return cp_callback


model = create_model(NUM_CLASSES)


def print_epoch_results(history):
    """
    Форматирует и выводит результаты обучения за каждую эпоху в виде таблицы.

    :param: History history: История обучения модели, возвращаемая методом `fit`.
    """
    epochs = np.arange(1, len(history.history['loss']) + 1)
    train_loss = np.array(history.history['loss'])
    train_accuracy = np.array(history.history['accuracy'])
    val_loss = np.array(history.history['val_loss'])
    val_accuracy = np.array(history.history['val_accuracy'])
    table = PrettyTable()
    table.field_names = ["Эпоха", "Потери (Train)", "Точность (Train)", "Потери (Val)", "Точность (Val)"]
    for epoch, t_loss, t_acc, v_loss, v_acc in zip(epochs, train_loss, train_accuracy, val_loss, val_accuracy):
        table.add_row([epoch, f"{t_loss:.4f}", f"{t_acc:.4f}", f"{v_loss:.4f}", f"{v_acc:.4f}"])
    print(table)


if latest_checkpoint:
    try:
        model.load_weights(latest_checkpoint)
        print("Веса успешно загружены.")
    except Exception as e:
        print(f"Ошибка при загрузке чекпоинта: {e}")

history = model.fit(create_generation_for_training(BATCH_SIZE, IMG_SHAPE, dir),
                    epochs=EPOCHS,
                    validation_data=validation(BATCH_SIZE, IMG_SHAPE, dir),
                    callbacks=[PlotLossesKeras(), create_checkpoint(checkpoint_path)])
model.save("emotion_model_learned.h5")

print_epoch_results(history)
