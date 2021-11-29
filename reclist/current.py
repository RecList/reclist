class Current():
    def __init__(self):
        self._report_path = None

    @property
    def report_path(self):
        return self._report_path


current = Current()
