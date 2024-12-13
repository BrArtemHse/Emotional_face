import json
"""
Этот код сохраняет словарь, содержащий названия эмоций и соответствующие ссылки на плейлисты, в файл формата JSON.
"""
link_on_playlist = {'Anger': 'https://vk.com/music/curator/mzk/playlists?z=audio_playlist-34384434_84577012',
                    'Contempt': 'https://vk.com/music/curator/mzk/playlists?z=audio_playlist-34384434_84577125',
                    'Disgust': 'https://vk.com/music/curator/mzk/playlists?z=audio_playlist-34384434_84576947',
                    'Fear': 'https://vk.com/music/curator/mzk/playlists?z=audio_playlist-34384434_84577148',
                    'Happy': 'https://vk.com/music/curator/mzk/playlists?z=audio_playlist-34384434_84577075',
                    'Neutreal': 'https://vk.com/music/curator/mzk/playlists?z=audio_playlist-34384434_84577345',
                    'Sad': 'https://vk.com/music/curator/mzk/playlists?z=audio_playlist-34384434_84576958',
                    'Surprised': 'https://vk.com/music/curator/mzk/playlists?z=audio_playlist-34384434_84577343'}

with open("link_for_playlist.json", "w", encoding="utf-8") as json_file:
    json.dump(link_on_playlist, json_file, ensure_ascii=False, indent=4)
