import sys
import time
from multiprocessing import RawArray, Process, Lock, Pipe, Event
from socket import socket, AF_INET, SOCK_STREAM
from enum import Enum

import numpy as np
import pyorbbecsdk as ob

from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QApplication, QMainWindow, QTableWidgetItem
from PyQt5 import QtCore, QtGui, QtWidgets

from robocup.network import DataType, pack_data
from robocup.camlxz import Ui_Form
from robocup.paru import Paru
from ultralytics.engine.results import Boxes

# [!] 轮次
ROUND = 2

# [!] 单个场景识别时长 (秒)
#  ** 第 1 轮总时最短 20s，最长 50s
#  ** 第 2 轮总时最短 70s，最长 150s，要求综合得分超过 30%
# 本时间应在调试时决定，适当调小
STAGE_TIME = 20

# [!] 相机云台旋转延时 (毫秒)
# 调试时决定
ROTATING_TIMEOUT = 3000

# [!] 每类物品最大数量
#  ** "同类物品最多为 5 个"
# 依据情况，可适当调低，去年参考值为 2
RESTRICT_NUM = 3

# [!] 裁判盒参数
SERVER_IP = '127.0.0.1'
SERVER_PORT = 6666

# 报名单位英文缩写-队伍名英文缩写
#  Wupin Detection with Neural Model
RESULT_FILE_PREFIX = "xjtu-wdnm"

# 结果目录
RESULT_PATH = "./result_r"

# frame 宽高
FRAME_W = 640
FRAME_H = 480

class Status(Enum):
    LOADING = 0
    READY = 1
    DETECTING = 2
    TURNING = 3
    FINISHED = 4

def get_config(pipeline):
    config = ob.Config()
    profile_list = pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR)
    try:
        color_profile = profile_list.get_video_stream_profile(640, 0, ob.OBFormat.RGB, 30)
    except Exception as e:
        print(e)
        color_profile = profile_list.get_default_video_stream_profile()
    config.enable_stream(color_profile)
    return config

def frame_to_image(frame):
    width = frame.get_width()
    height = frame.get_height()
    color_format = frame.get_format()
    data = np.asanyarray(frame.get_data())
    assert color_format == ob.OBFormat.RGB, "Camera color format should be RGB"
    image = np.resize(data, (height, width, 3))
    return image

class Ui_MainWindow(QMainWindow, Ui_Form):
    def __init__(self, parent=None):
        super(Ui_MainWindow, self).__init__(parent)

        self.setupUi(self)
        self.set_status(Status.LOADING)
        self.button_open_camera.setEnabled(False)

        self.socket = socket(AF_INET, SOCK_STREAM)

        # shared memory
        self.detect_proc_shm = RawArray('d', FRAME_H * FRAME_W * 3)
        self.buffer = np.frombuffer(self.detect_proc_shm, dtype=np.float64)
        # pipe for result label_dict_tx负责写入
        self.label_dict_rx, label_dict_tx = Pipe(False)
        # lock for sync
        self.detect_proc_lock = Lock()
        # detect process ready event
        ready_ev = Event()
        self.detect_ready = MsgAgent(ready_ev)
        self.detect_ready.connect(self.handle_detect_ready)
        # detect process sync event
        self.sync_ev = Event()
        # start slowly
        self.detect_proc = Process(target=detect_worker,
                                   args=(self.detect_proc_shm, label_dict_tx,
                                         self.detect_proc_lock, ready_ev, self.sync_ev))
        self.detect_proc.daemon = True
        self.detect_proc.start()
        self.detect_ready.start()

        self.timer_update = QtCore.QTimer()
        self.timer_turning = QtCore.QTimer()

        self.init_slots()

    def handle_detect_ready(self):
        self.set_status(Status.READY)
        self.button_open_camera.setEnabled(True)

    def init_slots(self):
        self.button_open_camera.clicked.connect(self.button_open_camera_clicked)
        self.timer_update.timeout.connect(self.update_ui_result)
        self.timer_turning.timeout.connect(self.restore_detecting)

    def task_finished(self):
        self.set_status(Status.FINISHED)
        # self.timer_update.stop()
        if self.detect_proc.is_alive():
            self.detect_proc.terminate()

    def button_open_camera_clicked(self):
        if self.status == Status.DETECTING:
            return
        self.set_status(Status.DETECTING)
        self.button_open_camera.setEnabled(False)

        self.sync_ev.set()
        print("worker started")
        try:
            self.socket.connect((SERVER_IP, SERVER_PORT))
            self.socket.send(pack_data(DataType.TEAM_ID, "xjtu"))
        except:
            pass
        self.timer_update.start(200)

    def restore_detecting(self):
        self.timer_turning.stop()
        self.label_turning.setVisible(False)
        self.set_status(Status.DETECTING)
        # WARN: `is_set()` is almost as costly as `set()`
        assert not self.sync_ev.is_set(), "sync event should not be set before restoring"
        self.sync_ev.set()

    def update_ui_result(self):
        if self.status == Status.TURNING:
            return

        if self.label_dict_rx.poll():
            try:
                label_dict = self.label_dict_rx.recv()
                if label_dict is None:
                    # request rotating
                    self.set_status(Status.TURNING)
                    self.label_turning.setVisible(True)
                    self.socket.send(pack_data(DataType.REQUEST_ROT, "0000"))
                    self.timer_turning.start(ROTATING_TIMEOUT)
                    return
                else:
                    # collect result
                    self.timer_update.stop()
                    self.label_camera.setPixmap(self.black_pixmap)
                    self.update_table(label_dict)
                    self.save_and_send_result(label_dict)
                    self.task_finished()
                    return
            except EOFError:
                print("detect process finished")
                self.timer_update.stop()

        self.detect_proc_lock.acquire(timeout=100)
        img = self.buffer.reshape(FRAME_H, FRAME_W, 3)
        self.detect_proc_lock.release()
        showImage = QtGui.QImage(img.astype(np.uint8), img.shape[1], \
                                 img.shape[0], QtGui.QImage.Format_RGB888)
        self.label_camera.setPixmap(QtGui.QPixmap.fromImage(showImage))

    def update_table(self, label_dict):
        self.table_result.clearContents()
        self.table_result.setRowCount(len(label_dict))
        for row, (name, num) in enumerate(label_dict.items()):
            self.table_result.setItem(row, 0, QTableWidgetItem(name))
            self.table_result.setItem(row, 1, QTableWidgetItem(str(num)))

    def serialize_result(self, label_dict):# 格式化传输信息
        EOL = "\r"
        result = f"START{EOL}"
        for name, num in label_dict.items():
            result += f"Goal_ID={name};Num={num}{EOL}"
        result += "END"
        return result

    def save_and_send_result(self, label_dict):
        res = self.serialize_result(label_dict)
        with open(f"{RESULT_PATH}/{RESULT_FILE_PREFIX}-R{ROUND}.txt", "w") as f:
            f.write(res)
        self.socket.send(pack_data(DataType.RESULT, res))

    def set_status(self, status):
        if status == Status.LOADING:
            text = "正在初始化"
        elif status == Status.READY:
            text = "就绪（空闲）"
        elif status == Status.DETECTING:
            text = "识别中"
        elif status == Status.TURNING:
            text = "转动中"
        elif status == Status.FINISHED:
            text = "结束"
        else:
            raise ValueError("非法状态")

        self.status = status
        self.label_status.setText(text)

class MsgAgent(QThread):
    sig_ev = pyqtSignal(bool)

    def __init__(self, ev):
        super().__init__()
        self.ev = ev

    def connect(self, f):
        self.sig_ev.connect(f)

    def run(self):
        self.ev.wait()
        self.sig_ev.emit(True)


def detect_worker(shared_buffer, label_dict_tx, lock, ready_ev, sync_ev):

    def wait_and_clear():
        # 这是为了等待相机旋转
        sync_ev.wait()
        sync_ev.clear()

    # camera context setup
    context = ob.Context()
    context.set_logger_to_console(ob.OBLogLevel.WARNING)
    context.set_logger_to_file(ob.OBLogLevel.DEBUG, "orbbecsdk.log")

    # device setup
    device_list = context.query_devices()
    if device_list is None or device_list.get_count() <= 0:
        raise IOError("no available device")
    device = device_list.get_device_by_index(0)
    if device.is_property_supported(ob.OBPropertyID.OB_PROP_COLOR_AUTO_EXPOSURE_BOOL,
                                    ob.OBPermissionType.PERMISSION_READ_WRITE):
        device.set_bool_property(ob.OBPropertyID.OB_PROP_COLOR_AUTO_EXPOSURE_BOOL, True)

    # pipeline & config construct
    pipeline = ob.Pipeline(device)
    config = get_config(pipeline)

    # shared buffer [important]
    # - should not be accessed without acquiring lock
    # - shape = (H*W*C)
    buffer = np.frombuffer(shared_buffer, dtype=np.float64)

    # load model & warm up
    model =Paru('./weights/yolov6s.pt', './formats/coco.yaml')
    print("warming up")
    model.detect_image(np.zeros(shape=(FRAME_H, FRAME_W, 3), dtype=np.uint8), draw_img=False)

    # process-local state variables
    class_set = set()
    object_counter = dict()


    # todo: 增强稳定性

    def predict(frame):
        image = frame_to_image(frame)

        results,detected_imgs = model.detect_image(np.asarray(image))
        result=results[0]
        result[0]=detected_imgs[0]

        boxes=result.boxes
        boxes_num=len(boxes.cls)

        for i in range(boxes_num):
            nameOfBox=model.class_names[i]
            if nameOfBox not in class_set:
                class_set.add(nameOfBox)
                object_counter[nameOfBox] = set()
            object_counter[nameOfBox].add(str(time.time_ns()))
            pass

        with lock:
            buffer[:] = detected_imgs.flatten()

    print("ready")
    # trigger ready event to update status label
    ready_ev.set() # 设置时间已经发生
    wait_and_clear()

    print("starting pipeline")
    pipeline.start(config)
    print("pipeline started")

    def next_stage(stage_time, final=False):
        t1 = time.perf_counter()
        while True:
            # print("loop")
            frames = pipeline.wait_for_frames(100)
            # print("frame recv")
            if frames is None:
                continue
            color_frame = frames.get_color_frame()
            if color_frame is None:
                continue
            t2 = time_start = time.perf_counter()

            if t2 - t1 > STAGE_TIME:
                model.reset_tracker()
                break

            # print("predict")
            predict(color_frame)

            time_end = time.perf_counter()
            fps = 1.0 / (time_end - time_start)
            print(f"current fps: {fps:.5f}")

        if final:
            print("task completed")
            label_dict = {}
            for name in class_set:
                label_dict[name] = min(len(object_counter[name]), RESTRICT_NUM)
            label_dict_tx.send(label_dict)
        else:
            print("next stage")
            label_dict_tx.send(None)

    if ROUND == 1:
        next_stage(STAGE_TIME, final=True)
    elif ROUND == 2:
        next_stage(STAGE_TIME)
        wait_and_clear() # for rotating
        next_stage(STAGE_TIME)
        wait_and_clear() # for rotating
        next_stage(STAGE_TIME, final=True)
    else:
        raise ValueError(f"`ROUND` should be either 1 or 2, got {ROUND}")

def main():
    app = QApplication(sys.argv)
    mywin = Ui_MainWindow()  # 实例化一个窗口小部件
    mywin.setWindowTitle('目标识别')  # 设置窗口标题
    mywin.show()  # 显示窗口
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
