class FacerecError(Exception):
    def __init__(self, msg):
        super().__init__(msg)


def facerec_assert(cond: bool, msg: str = ""):
    if not cond:
        raise FacerecError(msg)
