from deepface import DeepFace
import sys
import os
import json
from tensorflow.keras.models import load_model
from datetime import datetime
from PIL import Image
from email.mime.multipart import MIMEMultipart
from email import encoders
from email.mime.base import MIMEBase
import cv2
import numpy as np
import smtplib

"""

Этот скрипт позволяет анализировать изображения для определения эмоций, связывать их с плейлистами
и выполнять дополнительные функции, такие как сохранение результатов, отправка их на email или
обновление базы соответствий эмоций и плейлистов.

"""


def load_emotion_playlist_map(config_path="link_for_playlist.json"):
    """
    Загружает соответствие эмоций и плейлистов из конфигурационного файла.

    :param str config_path: Путь к JSON-файлу с конфигурацией (по умолчанию "link_for_playlist.json").
    :return: Словарь с эмоциями и соответствующими плейлистами.
    :rtype: dict.
    :raises FileNotFoundError: Если файл конфигурации отсутствует.

    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Файл конфигурации {config_path} не найден.")
    with open(config_path, 'r') as f:
        return json.load(f)


def validate_image_path(image_path):
    """
    Проверяет существование файла изображения и его допустимые форматы.

    :param str image_path: Путь к файлу изображения.
    :return: True, если файл существует и его формат поддерживается.
    :rtype: bool.
    :raises FileNotFoundError: Если файл не найден.
    :raises ValueError: Если формат файла неподдерживаемый.

    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Файл {image_path} не найден.")
    elif not image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise ValueError("Поддерживаются только файлы форматов .png, .jpg, .jpeg")
    else:
        print('Файл успешно найден')
        return True


def get_image_metadata(image_path):
    """
    Извлекает метаинформацию из изображения, включая размеры и дату последнего изменения.

    :param str image_path: Путь к файлу изображения.
    :return: Словарь с шириной, высотой и датой изменения файла.
    ::rtype: list.
    :raises Exception: Если произошла ошибка при извлечении метаданных.

    """
    try:
        with Image.open(image_path) as img:
            width, height = img.size
        modification_time = datetime.fromtimestamp(os.path.getmtime(image_path)).strftime('%Y-%m-%d %H:%M:%S')
        return {"width": width, "height": height, "modification_time": modification_time}
    except Exception as e:
        print(f"Ошибка при получении метаданных для {image_path}: {e}")
        return None


def analyze_image_with_self_education(image_path):
    """
    Анализирует изображение с использованием обученной модели для определения эмоции.

    :param str image_path: Путь к файлу изображения.
    :return: Строка с определённой эмоцией или None, если произошла ошибка.
    :rtype: string.
    :raises Exception: Если произошла ошибка при загрузке модели или анализе изображения.

    """
    try:
        model = load_model('emotion_model_learned.h5')
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (48, 48))
        image = image / 255.0
        image = np.expand_dims(image, axis=0)
        print(f"Начат анализ изображения: {image_path}")
        predictions = model.predict(image)
        emotion_dict = {
            0: 'angry',
            1: 'contempt',
            2: 'disgust',
            3: 'fear',
            4: 'happy',
            5: 'neutral',
            6: 'sad',
            7: 'surprise'
        }
        predicted_emotion = emotion_dict[np.argmax(predictions)]
        print(f"Эмоция успешно определена: {predicted_emotion}")
        return predicted_emotion
    except Exception as e:
        print(f"Ошибка при анализе изображения: {e}")
        return None


def analyze_image_with_module(image_path):
    """
    Анализирует изображение с использованием библиотеки DeepFace для определения эмоции.

    :param str image_path: Путь к файлу изображения.
    :return: Строка с определённой эмоцией или None, если произошла ошибка.
    :rtype: string.
    :raises Exception: Если произошла ошибка при анализе изображения.

        """
    try:
        print(f"Начат анализ изображения: {image_path}")
        analysis = DeepFace.analyze(img_path=image_path, actions=['emotion'])
        if isinstance(analysis, list):  # Если результат список
            emotion = analysis[0]['dominant_emotion']
        else:  # Если результат словарь
            emotion = analysis['dominant_emotion']
        print(f"Эмоция успешно определена: {emotion}")
        return emotion
    except Exception as e:
        print(f"Ошибка при анализе изображения: {e}")
        return None


def get_playlist_for_emotion(emotion, emotion_playlist_map):
    """
    Возвращает ссылку на плейлист, соответствующий заданной эмоции.

    :param str emotion: Название эмоции.
    :param dict emotion_playlist_map: Словарь с эмоциями и соответствующими плейлистами.
    :return: Ссылка на плейлист или None, если эмоция не найдена.
    :rtype: string.
    :raises Exception: Если произошла ошибка при поиске.

    """
    try:
        return emotion_playlist_map[emotion]
    except Exception as e:
        print(f'При поиске получена ошибка: {e}')
        return None


def process_image_list_with_deepface(image_list, emotion_playlist_map):
    """
    Обрабатывает список изображений с использованием DeepFace, определяет эмоции и возвращает результаты.

    :param str image_list: Список путей к изображениям.
    :param dict emotion_playlist_map: Словарь с эмоциями и соответствующими плейлистами.
    :return: Список кортежей с информацией о каждом изображении (путь, метаданные, эмоция, плейлист).
    :rtype: list.
    :raises Exception: Если произошла ошибка при обработке изображения.

    """
    results = []
    for image_path in image_list:
        try:
            validate_image_path(image_path)
            metadata = get_image_metadata(image_path)
            emotion = analyze_image_with_module(image_path)
            if emotion:
                playlist = get_playlist_for_emotion(emotion, emotion_playlist_map)
                print(emotion_playlist_map[emotion])
                results.append((image_path, metadata, emotion, playlist))
            else:
                print("Не удалось определить эмоцию")
        except Exception as e:
            print(f"Ошибка при обработке {image_path}: {e}")
            results.append((image_path, None, "Ошибка", str(e)))
    return results


def process_image_list_with_self_education_ns(image_list, emotion_playlist_map):
    """
    Обрабатывает список изображений с использованием пользовательской модели, определяет эмоции и возвращает результаты.

    :param str image_list: Список путей к изображениям.
    :param dict emotion_playlist_map: Словарь с эмоциями и соответствующими плейлистами.
    :return: Список кортежей с информацией о каждом изображении (путь, метаданные, эмоция, плейлист).
    :rtype: list.
    :raises Exception: Если произошла ошибка при обработке изображения.

    """
    results = []
    for image_path in image_list:
        try:
            validate_image_path(image_path)
            metadata = get_image_metadata(image_path)
            emotion = analyze_image_with_self_education(image_path)
            if emotion:
                playlist = get_playlist_for_emotion(emotion, emotion_playlist_map)
                results.append((image_path, metadata, emotion, playlist))
            else:
                print("Не удалось определить эмоцию")
        except Exception as e:
            print(f"Ошибка при обработке {image_path}: {e}")
            results.append((image_path, None, "Ошибка", str(e)))
    return results


def save_results_to_file(results, output_file="results.txt"):
    """
    Сохраняет результаты анализа в текстовый файл.

    :param list results: Список результатов анализа.
    :param output_file: Имя файла для сохранения (по умолчанию "results.txt").
    :rtype: file.
    :return: None.

    """
    with open(output_file, 'w') as f:
        for image, metadata, emotion, playlist in results:
            f.write(f"Изображение: {image}\n")
            if metadata:
                f.write(f"Размер: {metadata['width']}x{metadata['height']}\n")
                f.write(f"Дата изменения: {metadata['modification_time']}\n")
            f.write(f"Эмоция: {emotion}\n")
            f.write(f"Плейлист: {playlist}\n\n")
    print(f"Результаты сохранены в файл: {output_file}")


def filter_results_by_emotion(file_path, emotion_filter):
    """
    Фильтрует результаты анализа по заданной эмоции.

    :param str file_path: Путь к текстовому файлу с результатами анализа.
    :param str emotion_filter: Эмоция, по которой производится фильтрация.
    :return: Отфильтрованный список результатов.
    :rtype: list
    """
    filtered_results = []

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split(',')
            if len(parts) > 2 and parts[2] == emotion_filter:
                filtered_results.append(line.strip())

    return filtered_results


def filter_results_by_emotion(file_path, emotion_filter):
    """
    Фильтрует результаты анализа по заданной эмоции.

    :param str file_path: Путь к текстовому файлу с результатами анализа.
    :param str emotion_filter: Эмоция, по которой производится фильтрация.
    :return: Отфильтрованный список результатов.
    :rtype: list
    """
    filtered_results = []

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.split()
            print(parts)
            if len(parts) >= 2 and parts[1] == emotion_filter:
                filtered_results.append(parts[1])
    return filtered_results


def send_resuly_on_email(sender_email, sender_password, recipient_email):
    """
    Отправляет файл результатов на email.

    :param str sender_email: Email-адрес отправителя.
    :param str sender_password: Пароль отправителя.
    :param str recipient_email: Email-адрес получателя.
    :return: None.
    :rtype: None.
    :raises Exception: Если произошла ошибка при отправке письма.

    """
    try:
        # SMTP-сервер Яндекс.Почты
        smtp_server = "smtp.yandex.ru"
        smtp_port = 465

        # Формируем сообщение
        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = recipient_email
        msg["Subject"] = 'Playlist'
        attachment_path = 'results.txt'
        # Если указан файл для вложения
        if attachment_path and os.path.exists(attachment_path):
            with open(attachment_path, "rb") as attachment_file:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(attachment_file.read())
                encoders.encode_base64(part)
                part.add_header(
                    "Content-Disposition",
                    f"attachment; filename={os.path.basename(attachment_path)}",
                )
                msg.attach(part)

        # Устанавливаем соединение с сервером и отправляем письмо
        with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
            server.login(sender_email, sender_password)
            server.send_message(msg)
        print("Письмо успешно отправлено!")
    except Exception as e:
        print(f"Ошибка при отправке письма: {e}")


def update_emotion_playlist_map(config_path="link_for_playlist.json"):
    """
    Добавляет новую эмоцию и соответствующий плейлист в конфигурационный файл.

    :param str config_path: Путь к JSON-файлу с конфигурацией (по умолчанию "link_for_playlist.json").
    :return: None.
    :rtype: None.
    :raises Exception: Если произошла ошибка при обновлении конфигурации.

    """
    try:
        emotion_playlist_map = load_emotion_playlist_map(config_path)
        new_emotion = input("Введите новую эмоцию: ").strip()
        new_playlist = input("Введите ссылку на плейлист для этой эмоции: ").strip()
        if new_emotion in emotion_playlist_map:
            print("Эмоция уже существует в конфигурации.")
        else:
            emotion_playlist_map[new_emotion] = new_playlist
            with open(config_path, 'w') as f:
                json.dump(emotion_playlist_map, f, indent=4)
            print(f"Эмоция '{new_emotion}' добавлена с плейлистом: {new_playlist}")
    except Exception as e:
        print(f"Ошибка при обновлении конфигурации: {e}")


def interactive_mode(emotion_playlist_map):
    """
    Запускает интерактивный режим, где пользователь может анализировать изображения,
    добавлять эмоции или работать с результатами.

    :param dict emotion_playlist_map: Словарь с эмоциями и соответствующими плейлистами.
    :return: None.
    :rtype: None.

    """
    print("Добро пожаловать в интерактивный режим!")
    print('Выберите, какой нейросетью Вы хотите воспользоваться? DeepFace или Вашей?')
    answer = input()
    if answer == 'DeepFace':
        while True:
            print("\nВыберите действие:")
            print("1. Анализ одного изображения")
            print("2. Анализ нескольких изображений")
            print("3. Добавить новую эмоцию")
            print("4. Выход")

            choice = input("Ваш выбор: ").strip()
            if choice == "1":
                image_path = input("Введите путь к изображению: ").strip()
                emotion = analyze_image_with_module(image_path)
                print(get_playlist_for_emotion(emotion, emotion_playlist_map))
            elif choice == "2":
                image_paths = input("Введите пути к изображениям через запятую: ").strip().split(', ')
                results = process_image_list_with_deepface(image_paths, emotion_playlist_map)
                save_results_to_file(results)
            elif choice == "3":
                update_emotion_playlist_map()
            elif choice == "4":
                print("Выход из программы.")
                break
            else:
                print("Неверный выбор. Пожалуйста, попробуйте снова.")
        print('Хотите ли вы отправить файл по email?')
        answer = input()
        if answer == 'Да':
            print('Введите адрес почты вашей почты:', end=' ')
            sender_email = input()
            print('Введите пароль:', end=' ')
            sender_password = input()
            print('Адрес, по которому отправить результаты:', end=' ')
            recipient_email = input()
            send_resuly_on_email(sender_email, sender_password, recipient_email)
        else:
            pass
    else:
        while True:
            print("\nВыберите действие:")
            print("1. Анализ одного изображения")
            print("2. Анализ нескольких изображений")
            print("3. Добавить новую эмоцию")
            print("4. Выход")

            choice = input("Ваш выбор: ").strip()
            if choice == "1":
                image_path = input("Введите путь к изображению: ").strip()
                emotion = analyze_image_with_self_education(image_path)
                print(get_playlist_for_emotion(emotion, emotion_playlist_map))
            elif choice == "2":
                image_paths = input("Введите пути к изображениям через запятую: ").strip().split(', ')
                results = process_image_list_with_self_education_ns(image_paths, emotion_playlist_map)
                save_results_to_file(results)
            elif choice == "3":
                update_emotion_playlist_map()
            elif choice == "4":
                print("Выход из программы.")
                break
            else:
                print("Неверный выбор. Пожалуйста, попробуйте снова.")
        print('Хотите ли вы отправить файл по email?')
        answer = input()
        if answer == 'Да':
            print('Введите адрес почты вашей почты:', end=' ')
            sender_email = input()
            print('Введите пароль:', end=' ')
            sender_password = input()
            print('Адрес, по которому отправить результаты:', end=' ')
            recipient_email = input()
            send_resuly_on_email(sender_email, sender_password, recipient_email)
        else:
            pass


def main():
    """
    Основная функция программы. Запускает интерактивный режим или обрабатывает изображения, переданные через аргументы командной строки.
    """
    emotion_playlist_map = load_emotion_playlist_map()
    if len(sys.argv) == 1:
        interactive_mode(emotion_playlist_map)
    else:
        image_list = sys.argv[1:]
        results = process_image_list_with_deepface(image_list, emotion_playlist_map)
        save_results_to_file(results)


if __name__ == "__main__":
    main()
