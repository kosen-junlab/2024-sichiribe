from PyQt6.QtCore import pyqtSignal, QThread
from cores.exporter import Exporter
from cores.capture import FrameCapture
import logging
import numpy as np

class CaptureFeedWorker(QThread):
    cap_size = pyqtSignal(tuple)
    progress = pyqtSignal(np.ndarray)
    finished = pyqtSignal(np.ndarray)
    cancelled = pyqtSignal()
    error = pyqtSignal()

    def __init__(self, params, width, height):
        super().__init__()
        self.params = params
        self.width = width
        self.height = height
        self.logger = logging.getLogger('__main__').getChild(__name__)
        self._is_cancelled = False
        self._is_finished = False

    def run(self):
        fc = FrameCapture(self.params['device_num'])
        self.logger.info('Capture device(%s) loaded.' % self.params['device_num'])
        cap_width, cap_height = fc.set_cap_size(self.width, self.height)
        self.logger.debug('Capture size set to %d x %d' % (cap_width, cap_height))
        self.cap_size.emit((cap_width, cap_height))

        while True:
            if self._is_cancelled:
                self.cancelled.emit()
                break
            
            # フレームが正しくキャプチャされているかを確認
            frame = fc.capture()
            if frame is None:
                self.error.emit()
                break
            
            if self._is_finished:
                self.finished.emit(frame)
                break
            
            self.progress.emit(frame)
        fc.release()
            
    def stop(self):
        self.logger.info("Capture Feed stopping...") 
        self._is_finished = True  # 停止フラグを設定
        
    def cancel(self):
        self.logger.info("Capture Feed canceling...") 
        self._is_cancelled = True # 停止フラグを設定