import unittest

from emotional_face import analyze_image_with_module, get_image_metadata, process_image_list_with_deepface, \
    load_emotion_playlist_map, validate_image_path, analyze_image_with_self_education, \
    process_image_list_with_self_education_ns, get_playlist_for_emotion, filter_results_by_emotion

emotional_playlist_map = load_emotion_playlist_map()


def test_1():
    assert (analyze_image_with_module('test_set/test_1.jpeg') == 'angry')


def test_2():
    assert (analyze_image_with_module('test_set/test_2.mov') == None)


def test_3():
    assert (get_image_metadata('test_set/test_1.jpeg') == {"width": 900, "height": 600,
                                                           "modification_time": '2024-12-10 14:23:13'})


def test_4():
    assert (get_image_metadata('test_set/test_2.mov') == None)


def test_5():
    assert (process_image_list_with_deepface(['test_set/test_1.jpeg', 'test_set/test_2.mov'],
                                             emotional_playlist_map) == [('test_set/test_1.jpeg', {'height': 600,
                                                                                                   'modification_time': '2024-12-10 14:23:13',
                                                                                                   'width': 900},
                                                                          'angry',
                                                                          'https://vk.com/music/curator/mzk/playlists?z=audio_playlist-34384434_84577012'),
                                                                         ('test_set/test_2.mov', None, 'Ошибка',
                                                                          'Поддерживаются только файлы форматов .png, .jpg, .jpeg')])


def test_6():
    assert (validate_image_path('test_set/test_1.jpeg') == True)


def test_7():
    assert (analyze_image_with_self_education('test_set/test_1.jpeg') == 'disgust')


def test_8():
    assert (analyze_image_with_self_education('test_set/test_2.mov') == None)


def test_9():
    assert (process_image_list_with_self_education_ns(['test_set/test_1.jpeg', 'test_set/test_2.mov'],
                                                      emotional_playlist_map) == [
                ('test_set/test_1.jpeg', {'height': 600,
                                          'modification_time': '2024-12-10 14:23:13',
                                          'width': 900},
                 'disgust',
                 'https://vk.com/music/curator/mzk/playlists?z=audio_playlist-34384434_84576947'),
                ('test_set/test_2.mov', None, 'Ошибка',
                 'Поддерживаются только файлы форматов .png, .jpg, .jpeg')])


def test_10():
    assert (get_playlist_for_emotion('angry',
                                     emotional_playlist_map) == 'https://vk.com/music/curator/mzk/playlists?z=audio_playlist-34384434_84577012')


def test_11():
    assert (get_playlist_for_emotion('shame', emotional_playlist_map) == None)


def test_12():
    assert (filter_results_by_emotion('test_set/test1.txt', 'angry') == ['angry'])
