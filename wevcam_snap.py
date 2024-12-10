import cv2
import time

capture = cv2.VideoCapture(0)
img_counter = 0
frame_set = []
start_time = time.time()
print("Укажите эмоцию:", end='')
emotion = input()
while (True):
    ret, frame = capture.read()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    end_time = time.time()
    elapsed = end_time - start_time
    img_name = f"./data_set/{emotion}/data_{img_counter}.png"
    cv2.imwrite(img_name, frame)
    img_counter += 1
