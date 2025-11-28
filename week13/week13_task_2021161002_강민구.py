import cv2 as cv
import numpy as np
import threading, time
import SDcar
import sys
import tensorflow as tf
from tensorflow.keras.models import load_model

speed = 80
epsilon = 0.0001

# ---- 전역 상태 변수들 ----
enable_linetracing = False          # 지금은 사용 안 하지만 남겨둠
is_running = False                  # 스레드 종료 플래그
detec_on = False                    # t/r 로 ON/OFF
det_frame = None                    # Object detection 에서 사용할 최신 프레임
object_detected = False             # 감지 여부
enable_AIdrive = False              # e/w 로 AI 주행 ON/OFF


# ---------------- Object Detection Thread ----------------
def detec():
    global det_frame, detec_on, is_running, object_detected

    class_names = []
    with open('object_detection_classes_coco.txt', 'r') as f:
        class_names = f.read().strip().split('\n')

    COLORS = np.random.uniform(0, 255, size=(len(class_names), 3))

    net = cv.dnn.readNetFromTensorflow(
        'frozen_inference_graph.pb',
        'ssd_mobilenet_v2_coco_2018_03_29.pbtxt'
    )

    print('[detec] thread started.')

    while is_running:
        if detec_on and det_frame is not None:
            frame = det_frame.copy()
            h, w = frame.shape[:2]

            blob = cv.dnn.blobFromImage(
                image=frame, size=(300, 300), swapRB=True, crop=False
            )
            net.setInput(blob)
            out = net.forward()

            detected_any = False   # 이번 프레임에서 물체가 하나라도 잡혔는지

            for detection in out[0, 0, :, :]:
                confidence = float(detection[2])
                if confidence > 0.4:
                    detected_any = True       # 물체 감지!
                    class_id = int(detection[1])

                    x1 = int(detection[3] * w)
                    y1 = int(detection[4] * h)
                    x2 = int(detection[5] * w)
                    y2 = int(detection[6] * h)

                    if 1 <= class_id <= len(class_names):
                        class_name = class_names[class_id - 1]
                    else:
                        class_name = f'id_{class_id}'

                    color = COLORS[class_id % len(COLORS)]

                    cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv.putText(frame, class_name, (x1, y1 - 5),
                               cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # 감지 여부를 전역 플래그에 반영
            object_detected = detected_any
            if detected_any:
                print('[detec] object detected -> STOP')

            cv.imshow('detec', frame)
        else:
            # detection을 끄면 항상 False 로
            object_detected = False

        time.sleep(0.05)   # 너무 빡세게 돌지 않게 살짝 쉼

    print('[detec] thread finished.')


def func_thread():
    i = 0
    while True:
        # print("alive!!")
        time.sleep(1)
        i = i + 1
        if is_running is False:
            break


def key_cmd(which_key):
    print('which_key', which_key)
    is_exit = False
    global enable_AIdrive, detec_on   # ★ 두 전역 변수 모두 global

    if which_key & 0xFF == 184:
        print('up')
        car.motor_go(speed)
    elif which_key & 0xFF == 178:
        print('down')
        car.motor_back(speed)
    elif which_key & 0xFF == 180:
        print('left')
        car.motor_left(30)
    elif which_key & 0xFF == 182:
        print('right')
        car.motor_right(30)
    elif which_key & 0xFF == 181:
        car.motor_stop()
        enable_AIdrive = False
        print('stop')
    elif which_key & 0xFF == ord('q'):
        car.motor_stop()
        enable_AIdrive = False
        print('exit')
        print('enable_AIdrive: ', enable_AIdrive)
        is_exit = True
    elif which_key & 0xFF == ord('e'):
        enable_AIdrive = True
        print('enable_AIdrive: ', enable_AIdrive)
    elif which_key & 0xFF == ord('w'):
        enable_AIdrive = False
        car.motor_stop()
        print('enable_AIdrive 2: ', enable_AIdrive)
    # ---- Object Detection ON/OFF ----
    elif which_key & 0xFF == ord('t'):
        detec_on = True
        print('detection ON')
    elif which_key & 0xFF == ord('r'):
        detec_on = False
        print('detection OFF')
        # 필요하면 아래 주석 해제해서 창 닫기
        # cv.destroyWindow('detec')

    return is_exit


def detect_maskY_HSV(frame):
    crop_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    # sigmaX=0 으로 두고 borderType 기본값 사용
    crop_hsv = cv.GaussianBlur(crop_hsv, (5, 5), 0)
    # need to tune params
    mask_Y = cv.inRange(crop_hsv, (25, 50, 100), (35, 255, 255))
    return mask_Y


def detect_maskY_BGR(frame):
    B = frame[:, :, 0]
    G = frame[:, :, 1]
    R = frame[:, :, 2]
    Y = np.zeros_like(G, np.uint8)
    # need to tune params
    Y = G * 0.5 + R * 0.5 - B * 0.7  # 연산 수행 시 float64로 바뀜
    Y = Y.astype(np.uint8)
    Y = cv.GaussianBlur(Y, (5, 5), 0)
    # need to tune params
    _, mask_Y = cv.threshold(Y, 100, 255, cv.THRESH_BINARY)
    return mask_Y


def line_tracing(cx):
    # 지금은 안 쓰고 있음 (AI 주행만 사용)
    global moment
    global v_x
    tolerance = 0.1
    diff = 0

    if moment[0] != 0 and moment[1] != 0 and moment[2] != 0:
        avg_m = np.mean(moment)
        diff = np.abs(avg_m - cx) / v_x

    if diff <= tolerance:
        moment[0] = moment[1]
        moment[1] = moment[2]
        moment[2] = cx
        print('cx : ', cx)
        if v_x_grid[2] <= cx < v_x_grid[4]:
            car.motor_go(speed)
            print('go')
        elif v_x_grid[3] >= cx:
            car.motor_left(30)
            print('turn left')
        elif v_x_grid[1] <= cx:
            car.motor_right(30)
            print('turn right')
        else:
            print("skip")
    else:
        car.motor_go(speed)
        print('go')
        moment = [0, 0, 0]


def show_grid(img):
    h, _, _ = img.shape
    for x in v_x_grid:
        cv.line(img, (x, 0), (x, h), (0, 255, 0), 1, cv.LINE_4)


def test_fun(model):
    camera = cv.VideoCapture(15)
    camera.set(cv.CAP_PROP_FRAME_WIDTH, v_x)
    camera.set(cv.CAP_PROP_FRAME_HEIGHT, v_y)
    ret, frame = camera.read()
    frame = cv.flip(frame, -1)
    cv.imshow('camera', frame)
    crop_img = frame[int(v_y / 2):, :]
    crop_img = cv.resize(crop_img, (200, 66))
    crop_img = np.expand_dims(crop_img, 0)
    a = model.predict(crop_img)
    print('okey, a: ', a)


def drive_AI(img):
    img = np.expand_dims(img, 0)
    res = model.predict(img)[0]
    steering_angle = np.argmax(np.array(res))
    print('steering_angle', steering_angle)
    if steering_angle == 0:
        print("go")
        speedSet = 60
        car.motor_go(speedSet)
    elif steering_angle == 1:
        print("left")
        speedSet = 20
        car.motor_left(speedSet)
    elif steering_angle == 2:
        print("right")
        speedSet = 20
        car.motor_right(speedSet)
    else:
        print("This cannot be entered")


def main():
    global det_frame
    camera = cv.VideoCapture(0)
    camera.set(cv.CAP_PROP_FRAME_WIDTH, v_x)
    camera.set(cv.CAP_PROP_FRAME_HEIGHT, v_y)

    try:
        while camera.isOpened():
            ret, frame = camera.read()
            if not ret:
                break

            frame = cv.flip(frame, -1)
            cv.imshow('camera', frame)

            # object detection 스레드용 프레임 공유
            det_frame = frame

            # image processing start here
            crop_img = frame[int(v_y / 2):, :]
            crop_img = cv.resize(crop_img, (200, 66))

            maskY = detect_maskY_HSV(crop_img)

            contours, _ = cv.findContours(maskY, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            print('len(contours), ', len(contours))

            if len(contours) > 0:
                c = max(contours, key=cv.contourArea)
                m = cv.moments(c)

                cx = int(m['m10'] / (m['m00'] + epsilon))
                cy = int(m['m01'] / (m['m00'] + epsilon))
                cv.circle(crop_img, (cx, cy), 3, (0, 0, 255), -1)
                cv.drawContours(crop_img, contours, -1, (0, 255, 0), 3)

                cv.putText(crop_img, str(cx), (10, 10),
                           cv.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0))

            # AI 주행 ON 이면 모델 기반 조향
            if enable_AIdrive is True and not (detec_on and object_detected):
                # object_detected 가 True 일 때는 stop 이 우선이라 조건 추가
                drive_AI(crop_img)

            # Object detection + 감지 시 STOP
            if detec_on and object_detected:
                car.motor_stop()
                cv.putText(crop_img, 'STOP', (10, 40),
                           cv.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 2)

            show_grid(crop_img)
            cv.imshow('crop_img ', cv.resize(crop_img, dsize=(0, 0), fx=2, fy=2))

            # image processing end here
            is_exit = False
            which_key = cv.waitKey(20)
            if which_key > 0:
                is_exit = key_cmd(which_key)
            if is_exit is True:
                cv.destroyAllWindows()
                break

    except Exception as e:
        exception_type, exception_object, exception_traceback = sys.exc_info()
        filename = exception_traceback.tb_frame.f_code.co_filename
        line_number = exception_traceback.tb_lineno

        print("Exception type: ", exception_type)
        print("File name: ", filename)
        print("Line number: ", line_number)

        global is_running
        is_running = False


if __name__ == '__main__':

    v_x = 320
    v_y = 240
    v_x_grid = [int(v_x * i / 10) for i in range(1, 10)]
    print(v_x_grid)
    moment = np.array([0, 0, 0])

    model_path = 'lane_navigation_20251124_0525.h5'
    # model_path = 'my_checkpoint/lane_navigation_ce_stable.h5'     # tf1.15
    # model_path = 'my_checkpoint/lane_navigation_20221115_0808.h5'  # tf1.15
    # model_path = 'my_checkpoint/lane_navigation_20221115_0815.h5'  # tf2.2

    model = load_model(model_path)
    # print('id', id(model))
    # print(model.summary())

    car = SDcar.Drive()

    # 전역 상태 초기화
    is_running = True
    enable_AIdrive = False
    detec_on = False
    object_detected = False

    # 필요하면 heartbeat thread 도 다시 살릴 수 있음
    # t_task = threading.Thread(target=func_thread, daemon=True)
    # t_task.start()

    # Object detection thread 시작 (★ is_running True 이후!)
    t_det = threading.Thread(target=detec, daemon=True)
    t_det.start()

    main()

    is_running = False
    car.clean_GPIO()
    print('end vis')
