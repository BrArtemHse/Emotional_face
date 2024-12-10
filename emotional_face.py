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


def load_emotion_playlist_map(config_path="link_for_playlist.json"):
    """
    Загружает соответствие эмоций и плейлистов из конфигурационного файла.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Файл конфигурации {config_path} не найден.")
    with open(config_path, 'r') as f:
        return json.load(f)


def validate_image_path(image_path):
    """
    Проверяет существование файла и допустимые форматы.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Файл {image_path} не найден.")
    elif not image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise ValueError("Поддерживаются только файлы форматов .png, .jpg, .jpeg")
    else:
        return print('Файл успешно найден')


def get_image_metadata(image_path):
    """
    Извлекает метаинформацию из изображения.
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
    Анализирует изображение с помощью обученной модели, определяет эмоцию и возвращает её.
    """
    try:
        model = load_model('Logic_Model.h5')
        emotion_dict = {
            0: 'anger',
            1: 'contempt',
            2: 'disgust',
            3: 'fear',
            4: 'happy',
            5: 'neutral',
            6: 'sad',
            7: 'surprise'
        }
        image = cv2.imread(image_path)
        print(f"Начат анализ изображения: {image_path}")
        resized_image = cv2.resize(image, (128, 128))  # Размер, ожидаемый моделью
        resized_image = resized_image.astype('float32') / 255.0  # Нормализация
        resized_image = np.expand_dims(resized_image, axis=0)
        prediction = model.predict(resized_image)
        predicted_index = np.argmax(prediction)  # Индекс с наибольшей вероятностью
        predicted_emotion = emotion_dict[predicted_index]
        print(f"Эмоция успешно определена: {predicted_emotion}")
        return predicted_emotion
    except Exception as e:
        print(f"Ошибка при анализе изображения: {e}")
        return None


def analyze_image_with_module(image_path):
    """
        Анализирует изображение с помощью DeepFace, определяет эмоцию и возвращает её.
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
    Возвращает ссылку на плейлист, соответствующий эмоции.
    """
    try:
        return emotion_playlist_map[emotion]
    except Exception as e:
        print(f'При поиске получена ошибка: {e}')
        return None


def process_image_list_with_deepface(image_list, emotion_playlist_map):
    """
    Обрабатывает список изображений, анализирует каждое и выводит соответствующие плейлисты c deepface.
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
    Обрабатывает список изображений, анализирует каждое и выводит соответствующие плейлисты с self-made ns
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
    Сохраняет результаты обработки в текстовый файл.
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


def filter_results_by_emotion(results, emotion_filter):
    """
    Фильтрует результаты анализа по заданной эмоции.
    """
    filtered_results = [res for res in results if res[2] == emotion_filter]
    return filtered_results


def send_resuly_on_email(sender_email, sender_password, recipient_email):
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
    Позволяет добавить новую эмоцию и её плейлист в файл конфигурации.
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
    Запускает интерактивный режим для пользователя.
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
            elif choice == "3":
                update_emotion_playlist_map()
            elif choice == "4":
                print("Выход из программы.")
                break
            else:
                print("Неверный выбор. Пожалуйста, попробуйте снова.")
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
            elif choice == "3":
                update_emotion_playlist_map()
            elif choice == "4":
                print("Выход из программы.")
                break
            else:
                print("Неверный выбор. Пожалуйста, попробуйте снова.")


def main():
    """
    Основная функция программы.
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
