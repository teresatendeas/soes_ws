class PumpController:
    def __init__(self, on_cb, off_cb):
        self.on_cb = on_cb
        self.off_cb = off_cb
        self._is_on = False

    def start(self, duty: float = 1.0, duration_s: float = 0.0):
        self._is_on = True
        self.on_cb(duty, duration_s)

    def stop(self):
        if self._is_on:
            self._is_on = False
            self.off_cb()
