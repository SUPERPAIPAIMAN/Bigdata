import sys
import os
import av
import pyaudio
import threading
import time
import numpy as np
import pysrt
from PIL import Image, ImageDraw, ImageFont
import cv2  # 用于调整视频分辨率和渲染字幕
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QSlider, QVBoxLayout,
    QWidget, QFileDialog, QLabel, QListWidget, QHBoxLayout, QMessageBox, QSplitter, QSizePolicy, QComboBox, QMenu, QInputDialog
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, pyqtSlot, QPoint
from PyQt5.QtGui import QImage, QPixmap
from queue import Queue, Empty
from datetime import datetime
from urllib.parse import urlparse

# 映射 PyAV 音频格式到 PyAudio 格式
AUDIO_FORMAT_MAPPING = {
    'flt': pyaudio.paFloat32,
    'fltp': pyaudio.paFloat32,
    's16': pyaudio.paInt16,
    's32': pyaudio.paInt32,
    'dbl': pyaudio.paFloat32,  # PyAudio 没有 paDouble32，使用 paFloat32 代替
    # 根据需要添加更多格式映射
}

class SeekSlider(QSlider):
    """
    自定义的 QSlider，用于捕捉鼠标按下和释放事件。
    """
    mousePressed = pyqtSignal()
    mouseReleased = pyqtSignal()

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        self.mousePressed.emit()

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        self.mouseReleased.emit()

class AudioPlayer(QThread):
    errorOccurred = pyqtSignal(str)
    startedPlaying = pyqtSignal()  # 新增信号，表示音频播放器已启动

    def __init__(self, audio_queue, original_rate, speed, volume=1.0):
        super().__init__()
        self.audio_queue = audio_queue
        self.original_rate = original_rate
        self.speed = speed
        self.volume = volume
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()  # Not paused initially
        self.lock = threading.Lock()
        # 将重采样率设置为 original_rate / speed
        self.resampler_rate = max(1, int(original_rate / self.speed))
        self.resampler = av.audio.resampler.AudioResampler(
            format='flt',
            layout='stereo',
            rate=self.resampler_rate
        )

    def run(self):
        try:
            p = pyaudio.PyAudio()
            audio_format_str = 'flt'
            if audio_format_str not in AUDIO_FORMAT_MAPPING:
                self.errorOccurred.emit(f"不支持的目标音频格式: {audio_format_str}")
                return
            audio_format = AUDIO_FORMAT_MAPPING[audio_format_str]
            audio_channels = 2  # stereo
            with self.lock:
                audio_rate = self.resampler_rate
                print(f"[AudioPlayer] 音频采样率: {audio_rate}, 通道数: {audio_channels}, 格式: {audio_format_str}")
                stream = p.open(format=audio_format,
                                channels=audio_channels,
                                rate=audio_rate,
                                output=True)

            self.startedPlaying.emit()  # 发射启动信号

            while not self._stop_event.is_set():
                self._pause_event.wait()  # 等待未暂停
                try:
                    frame = self.audio_queue.get(timeout=0.1)  # 等待音频帧
                except Empty:
                    continue  # 没有音频帧，继续等待

                if frame is None:
                    # 收到结束信号
                    break

                try:
                    with self.lock:
                        resampled_frames = self.resampler.resample(frame)
                    for resampled_frame in resampled_frames:
                        audio_data = resampled_frame.to_ndarray()
                        audio_data = audio_data * self.volume
                        # 确保音频数据不超出 [-1.0, 1.0] 范围
                        audio_data = np.clip(audio_data, -1.0, 1.0)
                        audio_bytes = audio_data.astype(np.float32).tobytes()
                        stream.write(audio_bytes)
                except Exception as e:
                    self.errorOccurred.emit(f"音频处理出错: {e}")
                    break

            stream.stop_stream()
            stream.close()
            p.terminate()
            print("[AudioPlayer] 音频播放结束")
        except av.AVError as e:
            self.errorOccurred.emit(f"音频 AV 错误: {e}")
        except Exception as e:
            self.errorOccurred.emit(f"音频播放出错: {e}")

    def stop(self):
        self._stop_event.set()
        self.resume_playback()  # 确保线程从暂停中唤醒
        # 发送结束信号
        try:
            self.audio_queue.put_nowait(None)
        except:
            pass  # 如果队列已满或其他异常，忽略

    def pause(self):
        self._pause_event.clear()
        print("[AudioPlayer] 音频播放器已暂停")

    def resume_playback(self):
        self._pause_event.set()
        print("[AudioPlayer] 音频播放器已恢复")

    def set_speed(self, speed):
        with self.lock:
            self.speed = speed
            self.resampler_rate = max(1, int(self.original_rate / self.speed))
            self.resampler = av.audio.resampler.AudioResampler(
                format='flt',
                layout='stereo',
                rate=self.resampler_rate
            )
            print(f"[AudioPlayer] 音频播放器速度设置为: {self.speed}x, 新采样率: {self.resampler_rate}")

    def set_volume(self, volume):
        with self.lock:
            self.volume = volume
            print(f"[AudioPlayer] 音频播放器音量设置为: {self.volume}")

class VideoPlayerThread(QThread):
    frameReceived = pyqtSignal(QImage)
    finished = pyqtSignal()
    errorOccurred = pyqtSignal(str)
    positionChanged = pyqtSignal(float, float)  # current_time, duration
    durationChanged = pyqtSignal(float)  # 新增信号
    resolutionAvailable = pyqtSignal(int, int)  # 新增信号

    def __init__(self, filename, subtitles, volume=1.0, speed=1.0):
        super().__init__()
        self.filename = filename
        self.subtitles = subtitles  # 字幕列表
        self.volume = volume
        self.speed = speed
        self.target_resolution = None  # 新增：目标分辨率
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()  # Not paused initially
        self.current_time = 0.0
        self.duration = 0.0
        self.audio_player = None
        self.container = None
        self.audio_stream = None
        self.video_stream = None
        self.seek_lock = threading.Lock()  # 锁用于保护跳转操作
        self.seek_event = threading.Event()
        self.seek_target_time = 0.0
        self.demux_generator = None  # Demux 生成器

        self.brightness = 0  # 默认0
        self.contrast = 1.0  # 默认1.0

    def run(self):
        print(f"[VideoPlayerThread] 开始播放: {self.filename}")
        try:
            # 增加缓冲区大小和其他网络优化参数
            if self.is_url(self.filename):
                # 对于URL，增加缓冲区大小
                options = {
                    'buffer_size': '10000000',  # 缓冲
                    'max_delay': '500000',     # 最大延迟
                }
                self.container = av.open(self.filename, 'r', options=options)
                print("[VideoPlayerThread] 使用URL缓冲选项打开容器")
            else:
                self.container = av.open(self.filename)
                print("[VideoPlayerThread] 使用本地文件打开容器")

            self.audio_stream = next((s for s in self.container.streams if s.type == 'audio'), None)
            self.video_stream = next((s for s in self.container.streams if s.type == 'video'), None)
            if not self.video_stream:
                self.errorOccurred.emit("未找到视频流")
                self.finished.emit()
                return

            # 获取源视频分辨率
            source_width = self.video_stream.codec_context.width
            source_height = self.video_stream.codec_context.height
            print(f"[VideoPlayerThread] 源分辨率: {source_width}x{source_height}")

            # 通过信号传递分辨率信息
            self.resolutionAvailable.emit(source_width, source_height)

            if self.audio_stream:
                original_rate = self.audio_stream.codec_context.rate
                # 增加音频队列大小
                self.audio_queue = Queue(maxsize=500)  # 原为100，增大为500
                self.audio_player = AudioPlayer(
                    audio_queue=self.audio_queue,
                    original_rate=original_rate,
                    speed=self.speed,
                    volume=self.volume
                )
                self.audio_player.errorOccurred.connect(self.errorOccurred)
                self.audio_player.startedPlaying.connect(self.on_audio_started)
                self.audio_player.start()

                # 等待 AudioPlayer 启动
                self.audio_player_started = False
            else:
                self.audio_player = None

            video_fps = self.video_stream.average_rate
            if video_fps is None:
                video_fps = av.Rational(25, 1)  # 默认25 FPS
            frame_time = float(video_fps.denominator) / video_fps.numerator
            print(f"[VideoPlayerThread] 视频帧率: {video_fps}, 帧时间: {frame_time}")
            self.duration = self.container.duration / av.time_base if self.container.duration else 0

            # 发送durationChanged信号
            self.durationChanged.emit(self.duration)
            self.positionChanged.emit(self.current_time, self.duration)  # 初始位置

            start_time = time.time()
            last_frame_time = start_time

            # 初始化 demux 生成器
            self.demux_generator = self.container.demux(self.audio_stream, self.video_stream)

            while not self._stop_event.is_set():
                # 检查是否有跳转请求
                if self.seek_event.is_set():
                    with self.seek_lock:
                        target_time = self.seek_target_time
                        print(f"[VideoPlayerThread] 开始跳转到 {target_time} 秒")
                        # 使用视频流的 time_base 进行时间转换
                        seek_ts = int(target_time / self.video_stream.time_base)
                        # 执行跳转
                        self.container.seek(seek_ts, any_frame=False, backward=True, stream=self.video_stream)
                        print(f"[VideoPlayerThread] 跳转到 {target_time} 秒 (时间戳: {seek_ts})")

                        # 重新初始化 demux_generator
                        self.demux_generator = self.container.demux(self.audio_stream, self.video_stream)

                        # 停止旧的 AudioPlayer
                        if self.audio_player:
                            self.audio_player.stop()
                            self.audio_player.wait()

                        # 创建新的队列和 AudioPlayer
                        self.audio_queue = Queue(maxsize=500)  # 新的队列实例
                        self.audio_player = AudioPlayer(
                            audio_queue=self.audio_queue,
                            original_rate=self.audio_stream.codec_context.rate,
                            speed=self.speed,
                            volume=self.volume
                        )
                        self.audio_player.errorOccurred.connect(self.errorOccurred)
                        self.audio_player.startedPlaying.connect(self.on_audio_started)
                        self.audio_player.start()

                        # 等待 AudioPlayer 启动
                        self.audio_player_started = False
                        while not self.audio_player_started:
                            time.sleep(0.01)

                        # 重置播放时间
                        self.current_time = target_time
                        self.positionChanged.emit(self.current_time, self.duration)

                        # 立即获取并显示该位置的帧
                        try:
                            packet = next(self.demux_generator)
                            for frame in packet.decode():
                                if packet.stream.type == 'video':
                                    img = frame.to_ndarray(format='rgb24').copy()

                                    # 调整分辨率
                                    if self.target_resolution:
                                        target_width, target_height = self.target_resolution
                                        img = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_AREA)
                                        print(f"[VideoPlayerThread] 图像已调整为: {target_width}x{target_height}")

                                    # 应用亮度和对比度调整
                                    img = self.apply_brightness_contrast(img)

                                    # 绘制字幕
                                    img = self.render_subtitle(img, frame.pts * frame.time_base)

                                    height, width, channel = img.shape
                                    bytes_per_line = 3 * width
                                    qimage = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888).copy()
                                    self.frameReceived.emit(qimage)
                                    self.current_time = frame.pts * frame.time_base
                                    self.positionChanged.emit(self.current_time, self.duration)
                                    break  # 只显示一帧
                        except StopIteration:
                            print("[VideoPlayerThread] demux_generator 已耗尽")
                        except av.AVError as e:
                            self.errorOccurred.emit(f"AV 错误: {e}")
                        except Exception as e:
                            self.errorOccurred.emit(f"解码包出错: {e}")

                        self.seek_event.clear()
                        print(f"[VideoPlayerThread] 跳转到 {target_time} 秒完成")

                try:
                    packet = next(self.demux_generator)
                except StopIteration:
                    print("[VideoPlayerThread] demux_generator 已耗尽")
                    break
                except av.AVError as e:
                    self.errorOccurred.emit(f"AV 错误: {e}")
                    break
                except Exception as e:
                    self.errorOccurred.emit(f"解码包出错: {e}")
                    break

                if self._stop_event.is_set():
                    print("[VideoPlayerThread] 收到停止信号")
                    break

                self._pause_event.wait()  # 等待未暂停

                for frame in packet.decode():
                    if self._stop_event.is_set():
                        print("[VideoPlayerThread] 收到停止信号")
                        break

                    self._pause_event.wait()

                    if packet.stream.type == 'video':
                        try:
                            img = frame.to_ndarray(format='rgb24').copy()  # 复制数据以确保数据安全

                            # 获取当前目标分辨率
                            if self.target_resolution:
                                target_width, target_height = self.target_resolution
                                # 使用 OpenCV 调整图像大小
                                img = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_AREA)
                                print(f"[VideoPlayerThread] 图像已调整为: {target_width}x{target_height}")

                            # 应用亮度和对比度调整
                            img = self.apply_brightness_contrast(img)

                            # 绘制字幕
                            img = self.render_subtitle(img, frame.pts * frame.time_base)

                            height, width, channel = img.shape
                            bytes_per_line = 3 * width
                            qimage = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888).copy()
                            self.frameReceived.emit(qimage)
                        except Exception as e:
                            self.errorOccurred.emit(f"图像处理出错: {e}")
                            self.stop()
                            break

                        self.current_time = frame.pts * frame.time_base
                        self.positionChanged.emit(self.current_time, self.duration)

                        # 控制播放速度和帧率
                        current_time_time = time.time()
                        elapsed = current_time_time - last_frame_time
                        desired = frame_time / self.speed
                        sleep_time = desired - elapsed
                        if sleep_time > 0:
                            time.sleep(sleep_time)
                        last_frame_time = time.time()

                        if self._stop_event.is_set():
                            break
                    elif packet.stream.type == 'audio' and self.audio_player:
                        try:
                            self.audio_queue.put(frame, timeout=1)  # 增加超时处理
                        except Exception as e:
                            self.errorOccurred.emit(f"音频帧入队出错: {e}")
                            self.stop()
                            break

            # 播放完毕，发送结束信号
            if self.audio_player:
                self.audio_player.stop()
                self.audio_player.wait()

            self.container.close()
            print("[VideoPlayerThread] 关闭 PyAV 容器")
        except av.AVError as e:
            self.errorOccurred.emit(f"AV 错误: {e}")
        except Exception as e:
            self.errorOccurred.emit(f"播放出错: {e}")
        self.finished.emit()
        print("[VideoPlayerThread] 播放线程结束")

    @pyqtSlot(float)
    def seek_to(self, target_time):
        """
        跳转到指定的 target_time（秒）
        """
        if not self.container or not self.video_stream:
            print("[VideoPlayerThread] 无法跳转，容器或视频流未打开")
            return

        try:
            print(f"[VideoPlayerThread] 接收到跳转请求: {target_time} 秒")
            self.seek_target_time = target_time
            self.seek_event.set()
        except Exception as e:
            self.errorOccurred.emit(f"跳转出错: {e}")
            print(f"[VideoPlayerThread] 跳转出错: {e}")
            self.stop()

    def pause(self):
        self._pause_event.clear()
        if self.audio_player:
            self.audio_player.pause()
        print("[VideoPlayerThread] 视频播放器已暂停")

    def resume(self):
        self._pause_event.set()
        if self.audio_player:
            self.audio_player.resume_playback()
        print("[VideoPlayerThread] 视频播放器已恢复")

    def stop(self):
        self._stop_event.set()
        self._pause_event.set()  # 确保线程从暂停中唤醒
        if self.audio_player:
            self.audio_player.stop()
            self.audio_player.wait()
        print("[VideoPlayerThread] 视频播放器已停止")

    def set_volume(self, volume):
        self.volume = volume
        if self.audio_player:
            self.audio_player.set_volume(volume)
        print(f"[VideoPlayerThread] 设置音量为: {self.volume}")

    def set_speed(self, speed):
        self.speed = speed
        if self.audio_player:
            self.audio_player.set_speed(speed)
        # 注意：调整速度需要重新设置重采样器和同步
        print(f"[VideoPlayerThread] 设置倍速为: {self.speed}x")

    @pyqtSlot(int, int)
    def set_target_resolution(self, width, height):
        """
        设置目标分辨率
        """
        with self.seek_lock:
            self.target_resolution = (width, height)
            print(f"[VideoPlayerThread] 目标分辨率设置为: {width}x{height}")

    def render_subtitle(self, img, current_time):
        """
        在视频帧上渲染字幕
        """
        subtitle_text = self.get_current_subtitle(current_time)
        if subtitle_text:
            try:
                print("[render_subtitle] 正在渲染字幕:", subtitle_text)

                # 将 OpenCV 图像（RGB）转换为 PIL 图像（RGBA）以支持透明度
                img_pil = Image.fromarray(img).convert("RGBA")
                draw = ImageDraw.Draw(img_pil)

                font_path = "SimHei.ttf"  # 替换为您的中文字体路径
                if not os.path.isfile(font_path):
                    self.errorOccurred.emit(f"字体文件不存在: {font_path}")
                    print(f"[render_subtitle] 字体文件不存在: {font_path}")
                    return img

                font_size = 24  # 根据需要调整字体大小
                font = ImageFont.truetype(font_path, font_size)
                print("[render_subtitle] 字体加载成功")

                # 使用 textbbox 计算文本大小
                bbox = draw.textbbox((0, 0), subtitle_text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                print(f"[render_subtitle] 文本大小: {text_width}x{text_height}")

                # 计算位置：居中底部
                x = (img_pil.width - text_width) / 2
                y = img_pil.height - text_height - 30  # 30 像素边距
                print(f"[render_subtitle] 文本位置: ({x}, {y})")

                # 添加半透明背景矩形
                rectangle_color = (0, 0, 0, 128)  # 半透明黑色
                margin = 10
                draw.rectangle(
                    [(x - margin, y - margin), (x + text_width + margin, y + text_height + margin)],
                    fill=rectangle_color
                )
                print("[render_subtitle] 绘制半透明背景矩形")

                # 绘制文字
                text_color = (255, 255, 255, 255)  # 白色，不透明
                draw.text((x, y), subtitle_text, font=font, fill=text_color)
                print("[render_subtitle] 绘制文字")

                # 转换回 OpenCV 图像（RGB）
                img_rgba = np.array(img_pil)
                img_rgb = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2RGB)

                print("[render_subtitle] 字幕渲染完成")
                return img_rgb
            except IOError:
                self.errorOccurred.emit(f"无法加载字体文件: {font_path}")
                print(f"[render_subtitle] 无法加载字体文件: {font_path}")
                return img
            except Exception as e:
                self.errorOccurred.emit(f"渲染字幕出错: {e}")
                print(f"[render_subtitle] 渲染字幕出错: {e}")
                return img
        return img

    def get_current_subtitle(self, current_time):
        """
        获取当前时间对应的字幕文本
        """
        for subtitle in self.subtitles:
            start = subtitle.start.ordinal / 1000.0  # 转换为秒
            end = subtitle.end.ordinal / 1000.0
            if start <= current_time <= end:
                return subtitle.text.replace('\n', ' ')
            elif current_time < start:
                break  # 因为字幕是按时间排序的，可以提前退出
        return ""

    @pyqtSlot(list)
    def update_subtitles(self, subtitles):
        """
        更新字幕列表
        """
        with self.seek_lock:
            self.subtitles = subtitles
            print("[VideoPlayerThread] 字幕已更新")

    def apply_brightness_contrast(self, img):
        """
        应用亮度和对比度调整到图像
        """
        try:
            # alpha: 对比度控制 (1.0-3.0)
            # beta: 亮度控制 (0-100)
            alpha = self.contrast  # 对比度
            beta = self.brightness  # 亮度

            # 使用OpenCV函数调整亮度和对比度
            adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
            return adjusted
        except Exception as e:
            self.errorOccurred.emit(f"亮度/对比度调整出错: {e}")
            return img

    def is_url(self, path):
        """
        判断给定的路径是否是一个URL
        """
        parsed = urlparse(path)
        return parsed.scheme in ('http', 'https', 'ftp')

    @pyqtSlot()
    def on_audio_started(self):
        """
        处理 AudioPlayer 启动信号
        """
        self.audio_player_started = True
        print("[VideoPlayerThread] AudioPlayer 已启动")

class VideoPlayer(QMainWindow):
    # 定义一个新的信号用于跳转
    seekRequested = pyqtSignal(float)
    subtitlesLoaded = pyqtSignal(list)  # 新增信号，用于传递字幕列表

    def __init__(self):
        super().__init__()
        self.setWindowTitle("优化后的视频播放器")
        self.setGeometry(100, 100, 1600, 900)  # 增大窗口初始尺寸

        self.initUI()
        self.playlist = []
        self.current_index = -1
        self.player = None
        self.subtitles = []  # 存储字幕条目
        self.current_subtitle = None  # 当前显示的字幕
        self.is_seeking = False  # 标志变量，检测用户是否正在拖动进度条
        self.current_frame = None  # 当前显示的帧

        # 连接 seekRequested 信号到 handleSeekRequest 方法
        self.seekRequested.connect(self.handleSeekRequest)
        # 连接字幕加载信号到 VideoPlayerThread
        self.subtitlesLoaded.connect(self.update_subtitles_in_thread)

    def initUI(self):
        # 视频显示标签
        self.videoLabel = QLabel(self)
        self.videoLabel.setAlignment(Qt.AlignCenter)
        self.videoLabel.setStyleSheet("background-color: black;")
        self.videoLabel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # 使用 QSizePolicy

        # 播放按钮
        self.playButton = QPushButton("播放", self)
        self.playButton.clicked.connect(self.playPause)
        self.playButton.setEnabled(False)

        # 停止按钮
        self.stopButton = QPushButton("停止", self)
        self.stopButton.clicked.connect(self.stop)
        self.stopButton.setEnabled(False)

        # 加载字幕按钮
        self.loadSubtitleButton = QPushButton("加载字幕", self)
        self.loadSubtitleButton.clicked.connect(self.loadSubtitles)
        self.loadSubtitleButton.setEnabled(False)

        # 截图按钮
        self.screenshotButton = QPushButton("截图", self)
        self.screenshotButton.clicked.connect(self.takeScreenshot)
        self.screenshotButton.setEnabled(False)

        # 音量滑块
        self.volumeSlider = QSlider(Qt.Horizontal, self)
        self.volumeSlider.setRange(0, 100)
        self.volumeSlider.setValue(50)
        self.volumeSlider.valueChanged.connect(self.setVolume)

        # 进度条旁边的时间标签
        self.currentTimeLabel = QLabel("00:00", self)
        self.currentTimeLabel.setFixedWidth(60)
        self.totalTimeLabel = QLabel("00:00", self)
        self.totalTimeLabel.setFixedWidth(60)

        # 进度条
        self.positionSlider = SeekSlider(Qt.Horizontal, self)
        self.positionSlider.setRange(0, 1000)
        # 连接自定义的鼠标事件信号
        self.positionSlider.mousePressed.connect(self.start_seeking)
        self.positionSlider.mouseReleased.connect(self.end_seeking)

        # 倍速滑块
        self.speedSlider = QSlider(Qt.Horizontal, self)
        self.speedSlider.setRange(50, 150)  # 0.5x 到 1.5x
        self.speedSlider.setValue(100)
        self.speedSlider.valueChanged.connect(self.setPlaybackRate)

        # 选择视频按钮
        self.openButton = QPushButton("选择视频", self)
        self.openButton.clicked.connect(self.openFiles)

        # **新增：加载URL按钮**
        self.loadURLButton = QPushButton("加载URL", self)
        self.loadURLButton.clicked.connect(self.loadURL)

        # 播放列表
        self.playlistWidget = QListWidget(self)
        self.playlistWidget.doubleClicked.connect(self.playSelected)
        self.playlistWidget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.playlistWidget.customContextMenuRequested.connect(self.openPlaylistContextMenu)

        # 分辨率选择下拉框
        self.resolutionComboBox = QComboBox(self)
        self.resolutionComboBox.setEnabled(False)  # 初始禁用，直到加载视频
        self.resolutionComboBox.currentIndexChanged.connect(self.changeResolution)

        # **新增：亮度滑块**
        self.brightnessSlider = QSlider(Qt.Horizontal, self)
        self.brightnessSlider.setRange(0, 200)  # 0 到 200，100为默认
        self.brightnessSlider.setValue(100)
        self.brightnessSlider.valueChanged.connect(self.setBrightness)

        # **新增：对比度滑块**
        self.contrastSlider = QSlider(Qt.Horizontal, self)
        self.contrastSlider.setRange(50, 150)  # 0.5 到 1.5
        self.contrastSlider.setValue(100)
        self.contrastSlider.valueChanged.connect(self.setContrast)

        # 控制按钮布局
        controlLayout = QHBoxLayout()
        controlLayout.addWidget(self.openButton)
        controlLayout.addWidget(self.loadURLButton)  # 添加加载URL按钮
        controlLayout.addWidget(self.playButton)
        controlLayout.addWidget(self.stopButton)
        controlLayout.addWidget(self.loadSubtitleButton)
        controlLayout.addWidget(self.screenshotButton)  # 添加截图按钮
        controlLayout.addWidget(QLabel("音量"))
        controlLayout.addWidget(self.volumeSlider)
        controlLayout.addWidget(QLabel("倍速"))
        controlLayout.addWidget(self.speedSlider)
        controlLayout.addWidget(QLabel("对比度"))  # 添加对比度标签
        controlLayout.addWidget(self.contrastSlider)  # 添加对比度滑块
        controlLayout.addWidget(QLabel("亮度"))  # 添加亮度标签
        controlLayout.addWidget(self.brightnessSlider)  # 添加亮度滑块
        controlLayout.addWidget(QLabel("分辨率"))  # 添加分辨率标签
        controlLayout.addWidget(self.resolutionComboBox)  # 添加分辨率选择控件

        # 进度条布局，包含当前时间、进度条和总时间
        progressLayout = QHBoxLayout()
        progressLayout.addWidget(self.currentTimeLabel)
        progressLayout.addWidget(self.positionSlider)
        progressLayout.addWidget(self.totalTimeLabel)

        # 使用水平 QSplitter 分割视频显示和播放列表
        splitter = QSplitter(Qt.Horizontal)

        # 左侧：视频显示、进度条和控制按钮
        leftLayout = QVBoxLayout()
        leftLayout.addWidget(self.videoLabel)
        leftLayout.addLayout(progressLayout)
        leftLayout.addLayout(controlLayout)
        leftWidget = QWidget()
        leftWidget.setLayout(leftLayout)
        splitter.addWidget(leftWidget)

        # 右侧：播放列表
        playlistLayout = QVBoxLayout()
        playlistLayout.addWidget(QLabel("播放列表"))
        playlistLayout.addWidget(self.playlistWidget)
        playlistWidget = QWidget()
        playlistWidget.setLayout(playlistLayout)
        splitter.addWidget(playlistWidget)

        # 设置伸缩因子，使左侧（视频）占据更多空间
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        # 主布局
        mainLayout = QVBoxLayout()
        mainLayout.addWidget(splitter)

        centralWidget = QWidget()
        centralWidget.setLayout(mainLayout)
        self.setCentralWidget(centralWidget)

        # 定时器更新进度条
        self.timer = QTimer()
        self.timer.setInterval(200)
        self.timer.timeout.connect(self.updatePosition)

        # 定时器用于在拖动时每0.2秒更新帧
        self.seekTimer = QTimer()
        self.seekTimer.setInterval(200)
        self.seekTimer.timeout.connect(self.updateSeekingFrame)

    def openPlaylistContextMenu(self, position):
        """
        打开播放列表的上下文菜单
        """
        item = self.playlistWidget.itemAt(position)
        if item:
            menu = QMenu()
            removeAction = menu.addAction("移除")
            action = menu.exec_(self.playlistWidget.mapToGlobal(position))
            if action == removeAction:
                self.removePlaylistItem(self.playlistWidget.row(item))

    def removePlaylistItem(self, index):
        """
        从播放列表中移除指定索引的项目
        """
        try:
            if 0 <= index < len(self.playlist):
                removed_video = self.playlist.pop(index)
                self.playlistWidget.takeItem(index)
                print(f"[VideoPlayer] 移除播放列表中的视频: {removed_video}")

                # 如果移除的是当前播放的视频
                if index == self.current_index:
                    self.stop()
                    if self.playlist:
                        # 如果还有其他视频，自动播放下一个或第一个
                        if index < len(self.playlist):
                            self.current_index = index
                        else:
                            self.current_index = len(self.playlist) - 1
                        self.startPlayback()
                    else:
                        self.current_index = -1
                elif index < self.current_index:
                    # 如果移除的是当前播放列表中当前索引之前的视频，调整当前索引
                    self.current_index -= 1
        except Exception as e:
            QMessageBox.critical(self, "错误", f"移除播放列表项出错: {e}")
            print(f"[VideoPlayer] 移除播放列表项出错: {e}")

    def populateResolutionOptions(self, width, height):
        """
        根据源视频分辨率，填充分辨率选择下拉框
        """
        try:
            self.resolutionComboBox.blockSignals(True)  # 临时阻塞信号，避免触发事件
            self.resolutionComboBox.clear()

            # 定义标准分辨率列表，从高到低
            standard_resolutions = [
                (1920, 1080),
                (1600, 900),
                (1366, 768),
                (1280, 720),
                (854, 480),
                (640, 360),
                (426, 240),
            ]

            # 过滤不大于源分辨率的选项
            available_resolutions = [
                f"{w}x{h}" for w, h in standard_resolutions if w <= width and h <= height
            ]

            # 如果源分辨率不在标准列表中，添加源分辨率作为选项
            if (width, height) not in standard_resolutions:
                available_resolutions.insert(0, f"{width}x{height}")

            self.resolutionComboBox.addItems(available_resolutions)
            self.resolutionComboBox.setCurrentText(f"{width}x{height}")  # 默认选中源分辨率
            self.resolutionComboBox.setEnabled(True)
            self.target_resolution = (width, height)  # 默认目标分辨率为源分辨率

            print(f"[VideoPlayer] 可用分辨率: {available_resolutions}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"填充分辨率选项出错: {e}")
            print(f"[VideoPlayer] 填充分辨率选项出错: {e}")
        finally:
            self.resolutionComboBox.blockSignals(False)  # 解除信号阻塞

    def changeResolution(self, index):
        """
        处理用户选择的分辨率更改
        """
        try:
            if not self.player:
                return

            resolution_text = self.resolutionComboBox.currentText()
            width, height = map(int, resolution_text.split('x'))
            self.player.set_target_resolution(width, height)
            print(f"[VideoPlayer] 用户选择的新分辨率: {width}x{height}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"更改分辨率出错: {e}")
            print(f"[VideoPlayer] 更改分辨率出错: {e}")

    def start_seeking(self):
        """用户开始拖动进度条时调用"""
        self.is_seeking = True
        if self.player:
            self.player.pause()
        self.seekTimer.start()
        print("[VideoPlayer] 用户开始拖动进度条，视频已暂停")

    def end_seeking(self):
        """用户释放进度条滑块时调用"""
        self.is_seeking = False
        self.seekTimer.stop()
        if self.player:
            target_time = (self.positionSlider.value() / 1000.0) * self.player.duration
            print(f"[VideoPlayer] 用户释放进度条，跳转到: {target_time} 秒")
            self.seekRequested.emit(target_time)  # 通过信号发出跳转请求
            self.player.resume()
        self.timer.start()
        print("[VideoPlayer] 视频已恢复播放")

    def updateSeekingFrame(self):
        if self.player and self.is_seeking:
            try:
                position = self.positionSlider.value()
                target_time = (position / 1000.0) * self.player.duration
                print(f"[VideoPlayer] 跳转到: {target_time} 秒 (拖动中)")
                self.seekRequested.emit(target_time)  # 通过信号发出跳转请求
            except Exception as e:
                QMessageBox.critical(self, "错误", f"更新寻帧出错: {e}")
                print(f"[VideoPlayer] 更新寻帧出错: {e}")

    def openFiles(self):
        try:
            files, _ = QFileDialog.getOpenFileNames(
                self, "选择视频文件", "", "视频文件 (*.mp4 *.avi *.mkv *.mov)"
            )
            if files:
                self.playlist.extend(files)
                self.playlistWidget.addItems([self.extract_filename(f) for f in files])
                self.playButton.setEnabled(True)
                self.stopButton.setEnabled(True)
                self.loadSubtitleButton.setEnabled(True)
                self.screenshotButton.setEnabled(True)  # 启用截图按钮
                self.resolutionComboBox.setEnabled(False)  # 重置分辨率选择
                self.resolutionComboBox.clear()
                if self.current_index == -1:
                    self.current_index = 0
                print(f"[VideoPlayer] 已添加 {len(files)} 个视频到播放列表")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"添加文件出错: {e}")
            print(f"[VideoPlayer] 添加文件出错: {e}")

    def loadSubtitles(self):
        try:
            subtitle_file, _ = QFileDialog.getOpenFileName(
                self, "选择字幕文件", "", "字幕文件 (*.srt)"
            )
            if subtitle_file:
                self.subtitles = pysrt.open(subtitle_file)
                print(f"[VideoPlayer] 加载字幕文件: {subtitle_file}")
                QMessageBox.information(self, "成功", "字幕文件加载成功！")
                # 发送字幕加载信号到 VideoPlayerThread，转换为列表
                self.subtitlesLoaded.emit(list(self.subtitles))
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载字幕文件出错: {e}")
            print(f"[VideoPlayer] 加载字幕文件出错: {e}")

    def playSelected(self):
        try:
            selected_items = self.playlistWidget.selectedIndexes()
            if selected_items:
                selected_row = selected_items[0].row()
                if selected_row != self.current_index:
                    self.current_index = selected_row
                    print(f"[VideoPlayer] 选择播放列表中的视频: 索引 {self.current_index}")
                    self.startPlayback()
                else:
                    print("[VideoPlayer] 选择的视频与当前播放的视频相同，无需重新播放")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"播放选择的视频出错: {e}")
            print(f"[VideoPlayer] 播放选择的视频出错: {e}")

    def playPause(self):
        try:
            if self.player:
                if self.playButton.text() == "暂停":
                    self.player.pause()
                    self.playButton.setText("播放")
                    print("[VideoPlayer] 暂停播放")
                    self.timer.stop()
                else:
                    self.player.resume()
                    self.playButton.setText("暂停")
                    print("[VideoPlayer] 恢复播放")
                    self.timer.start()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"播放/暂停出错: {e}")
            print(f"[VideoPlayer] 播放/暂停出错: {e}")

    def stop(self):
        try:
            if self.player:
                self.player.stop()
                self.player.wait()
                self.player = None
                self.playButton.setText("播放")
                self.timer.stop()
                self.seekTimer.stop()
                self.positionSlider.setValue(0)
                self.currentTimeLabel.setText("00:00")
                self.totalTimeLabel.setText("00:00")
                self.is_seeking = False
                self.resolutionComboBox.setEnabled(False)  # 重置分辨率选择
                self.resolutionComboBox.clear()
                print("[VideoPlayer] 停止播放")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"停止播放出错: {e}")
            print(f"[VideoPlayer] 停止播放出错: {e}")

    def setVolume(self, value):
        try:
            if self.player:
                self.player.set_volume(value / 100.0)
                print(f"[VideoPlayer] 设置音量: {value / 100.0}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"设置音量出错: {e}")
            print(f"[VideoPlayer] 设置音量出错: {e}")

    def setPlaybackRate(self, value):
        try:
            if self.player:
                new_speed = value / 100.0
                self.player.set_speed(new_speed)
                print(f"[VideoPlayer] 设置倍速: {new_speed}x")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"设置倍速出错: {e}")
            print(f"[VideoPlayer] 设置倍速出错: {e}")

    @pyqtSlot(float)
    def handleSeekRequest(self, target_time):
        """
        处理来自进度条的跳转请求
        """
        if self.player:
            print(f"[VideoPlayer] 发送跳转请求到播放线程: {target_time} 秒")
            self.player.seek_to(target_time)

    def updatePosition(self):
        try:
            if self.player and self.player.duration > 0:
                if not self.is_seeking:
                    pos = self.player.current_time / self.player.duration
                    self.positionSlider.setValue(int(pos * 1000))
                # 更新时间标签
                current_time_seconds = self.player.current_time
                total_time_seconds = self.player.duration
                self.currentTimeLabel.setText(self.format_time(current_time_seconds))
                self.totalTimeLabel.setText(self.format_time(total_time_seconds))
                print(f"[VideoPlayer] 当前播放位置: {self.player.current_time}/{self.player.duration}")
            if self.player and self.player.current_time >= self.player.duration:
                self.timer.stop()
                self.playButton.setText("播放")
                self.playNext()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"更新进度条出错: {e}")
            print(f"[VideoPlayer] 更新进度条出错: {e}")

    def format_time(self, seconds):
        """
        格式化时间为 HH:MM:SS 或 MM:SS
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        if hours > 0:
            return f"{hours:02}:{minutes:02}:{secs:02}"
        else:
            return f"{minutes:02}:{secs:02}"

    def playNext(self):
        try:
            self.current_index += 1
            if self.current_index < len(self.playlist):
                print(f"[VideoPlayer] 播放下一个视频: 索引 {self.current_index}")
                self.startPlayback()
            else:
                self.current_index = -1
                print("[VideoPlayer] 播放列表结束")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"播放下一个视频出错: {e}")
            print(f"[VideoPlayer] 播放下一个视频出错: {e}")

    def startPlayback(self):
        try:
            if self.player:
                print("[VideoPlayer] 停止当前播放线程")
                self.player.stop()
                self.player.wait()
                self.player = None

            if self.current_index < 0 or self.current_index >= len(self.playlist):
                QMessageBox.warning(self, "警告", "播放索引超出范围")
                print("[VideoPlayer] 播放索引超出范围")
                return

            video_path = self.playlist[self.current_index]
            print(f"[VideoPlayer] 开始播放: {video_path}")

            # 重置UI元素
            self.positionSlider.setValue(0)
            self.currentTimeLabel.setText("00:00")
            self.totalTimeLabel.setText("00:00")
            self.is_seeking = False
            self.resolutionComboBox.setEnabled(False)  # 重置分辨率选择
            self.resolutionComboBox.clear()

            self.player = VideoPlayerThread(
                filename=video_path,
                subtitles=self.subtitles,  # 传递字幕列表
                volume=self.volumeSlider.value() / 100.0,
                speed=self.speedSlider.value() / 100.0
            )
            self.player.frameReceived.connect(self.updateFrame)
            self.player.finished.connect(self.onPlaybackFinished)
            self.player.errorOccurred.connect(self.onError)
            self.player.positionChanged.connect(self.onPositionChanged)
            self.player.durationChanged.connect(self.onDurationChanged)  # 连接durationChanged信号
            self.player.resolutionAvailable.connect(self.populateResolutionOptions)  # 连接新信号
            # 连接字幕加载信号到 VideoPlayerThread
            self.subtitlesLoaded.connect(self.player.update_subtitles)

            self.player.start()
            self.playButton.setText("暂停")
            self.timer.start()
            print("[VideoPlayer] 播放已启动")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"启动播放出错: {e}")
            print(f"[VideoPlayer] 启动播放出错: {e}")

    def updateFrame(self, image):
        try:
            if not image.isNull():
                pix = QPixmap.fromImage(image)
                self.videoLabel.setPixmap(pix.scaled(
                    self.videoLabel.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
                self.current_frame = image  # 存储当前帧
        except Exception as e:
            QMessageBox.critical(self, "错误", f"更新视频帧出错: {e}")
            print(f"[VideoPlayer] 更新视频帧出错: {e}")

    def onPlaybackFinished(self):
        try:
            self.timer.stop()
            self.playButton.setText("播放")
            self.current_subtitle = None
            print("[VideoPlayer] 播放完成")
            self.playNext()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"播放完成处理出错: {e}")
            print(f"[VideoPlayer] 播放完成处理出错: {e}")

    def onError(self, message):
        try:
            QMessageBox.critical(self, "错误", message)
            print(f"[VideoPlayer] 错误: {message}")
            self.stop()
        except Exception as e:
            print(f"[VideoPlayer] 错误处理出错: {e}")

    def onPositionChanged(self, current_time, duration):
        try:
            if duration > 0:
                pos = current_time / duration
                if not self.is_seeking:
                    self.positionSlider.setValue(int(pos * 1000))
        except Exception as e:
            QMessageBox.critical(self, "错误", f"更新位置出错: {e}")
            print(f"[VideoPlayer] 更新位置出错: {e}")

    def onDurationChanged(self, duration):
        """
        处理 durationChanged 信号，更新总时长标签
        """
        try:
            self.totalTimeLabel.setText(self.format_time(duration))
            print(f"[VideoPlayer] 视频总时长更新为: {self.format_time(duration)}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"更新总时长出错: {e}")
            print(f"[VideoPlayer] 更新总时长出错: {e}")

    def handleSeekRequest(self, target_time):
        """
        处理来自进度条的跳转请求
        """
        if self.player:
            print(f"[VideoPlayer] 发送跳转请求到播放线程: {target_time} 秒")
            self.player.seek_to(target_time)

    def takeScreenshot(self):
        """
        截取当前帧并保存到项目根目录的 'screenshots' 文件夹
        """
        if self.current_frame:
            try:
                # 获取脚本所在目录
                script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
                # 定义截图保存目录
                screenshots_dir = os.path.join(script_dir, "screenshots")
                # 如果目录不存在，则创建
                if not os.path.exists(screenshots_dir):
                    os.makedirs(screenshots_dir)
                    print(f"[VideoPlayer] 创建截图目录: {screenshots_dir}")

                # 生成时间戳
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"screenshot_{timestamp}.png"
                filepath = os.path.join(screenshots_dir, filename)

                # 保存截图
                if self.current_frame.save(filepath):
                    QMessageBox.information(self, "成功", f"截图已保存到 {filepath}")
                    print(f"[VideoPlayer] 截图已保存到 {filepath}")
                else:
                    QMessageBox.critical(self, "错误", "保存截图失败")
                    print("[VideoPlayer] 保存截图失败")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"截图出错: {e}")
                print(f"[VideoPlayer] 截图出错: {e}")
        else:
            QMessageBox.warning(self, "警告", "当前没有可截图的帧")
            print("[VideoPlayer] 当前没有可截图的帧")

    def update_subtitles_in_thread(self, subtitles):
        """
        更新字幕到 VideoPlayerThread
        """
        if self.player:
            self.player.update_subtitles(subtitles)
            print("[VideoPlayer] 字幕已发送到播放线程")

    def closeEvent(self, event):
        try:
            if self.player:
                self.player.stop()
                self.player.wait()
            event.accept()
            print("[VideoPlayer] 程序关闭")
        except Exception as e:
            print(f"[VideoPlayer] 关闭程序时出错: {e}")
            event.accept()

    def extract_filename(self, filepath):
        if self.is_url(filepath):
            parsed_url = urlparse(filepath)
            return os.path.basename(parsed_url.path) or "URL视频"
        else:
            return os.path.basename(filepath)

    def is_url(self, path):
        """
        判断给定的路径是否是一个URL
        """
        parsed = urlparse(path)
        return parsed.scheme in ('http', 'https', 'ftp')

    # **新增：加载URL**
    def loadURL(self):
        try:
            url, ok = QInputDialog.getText(self, "加载视频URL", "请输入视频的URL：")
            if ok and url:
                if not self.is_url(url):
                    QMessageBox.critical(self, "错误", "请输入有效的URL（以http、https或ftp开头）")
                    print("[VideoPlayer] 无效的URL输入")
                    return
                self.playlist.append(url)
                self.playlistWidget.addItem(self.extract_filename(url))
                self.playButton.setEnabled(True)
                self.stopButton.setEnabled(True)
                self.loadSubtitleButton.setEnabled(True)
                self.screenshotButton.setEnabled(True)  # 启用截图按钮
                self.resolutionComboBox.setEnabled(False)  # 重置分辨率选择
                self.resolutionComboBox.clear()
                if self.current_index == -1:
                    self.current_index = 0
                print(f"[VideoPlayer] 已添加URL到播放列表: {url}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载URL出错: {e}")
            print(f"[VideoPlayer] 加载URL出错: {e}")

    def setBrightness(self, value):
        try:
            if self.player:
                # 将滑块值映射到亮度范围，假设滑块范围为0-200，映射到0到+100
                brightness = value / 2
                self.player.brightness = brightness
                print(f"[VideoPlayer] 设置亮度: {brightness}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"设置亮度出错: {e}")
            print(f"[VideoPlayer] 设置亮度出错: {e}")

    def setContrast(self, value):
        try:
            if self.player:
                # 将滑块值映射到对比度范围，假设滑块范围为50-150，映射到0.5-1.5
                contrast = value / 100.0
                self.player.contrast = contrast
                print(f"[VideoPlayer] 设置对比度: {contrast}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"设置对比度出错: {e}")
            print(f"[VideoPlayer] 设置对比度出错: {e}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    player = VideoPlayer()
    player.show()
    sys.exit(app.exec_())
