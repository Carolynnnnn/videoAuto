from PyQt5.QtCore import QThread, pyqtSignal

class WorkerThread(QThread):
    progress_signal = pyqtSignal(str)
    step_signal = pyqtSignal(int, str)   # (step_index, status: "active"|"done"|"error")
    finished_signal = pyqtSignal(bool, str)  # (success, result_path_or_error)
    percent_signal = pyqtSignal(int)

    def __init__(self, task_fn, *args, **kwargs):
        super().__init__()
        self.task_fn = task_fn
        self.args = args
        self.kwargs = kwargs

    def run(self):
        try:
            result = self.task_fn(*self.args, **self.kwargs)
            self.finished_signal.emit(True, str(result) if result else "")
        except Exception as e:
            import traceback
            self.finished_signal.emit(False, traceback.format_exc())
