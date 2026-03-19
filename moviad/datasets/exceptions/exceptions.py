class DatasetTooSmallToContaminateException(Exception):
    def __init__(self, message="The dataset is not large enough to be contaminated"):
        self.message = message
        super().__init__(self.message)