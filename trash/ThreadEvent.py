import time

class ThreadEvent():
    def __init__(self):
        self.__flag_count = 0
        self.__time = 0.0

    def set(self):
        if self.__flag_count > 0:
            self.__flag_count -= 1

    def clear(self):
        self.__flag_count = 0

    def wait(self, count, seconds):
        if type(count) == int:
            self.__flag_count = count
        else:
            raise TypeError
        while True:
            if self.__flag_count == 0:
                return True
            self.__time += 0.1
            if self.__time >= seconds:
                self.__time = 0.0
                return False
            time.sleep(0.1)

if __name__ == '__main__':
    event = ThreadEvent()
    print event.wait(1, 10)