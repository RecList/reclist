class Current:
    def __init__(self)-> None:
        self._report_path = None

    @property
    def report_path(self)->str:
        return self._report_path


current = Current()
