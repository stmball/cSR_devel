
from threading import Lock
import atexit
import sys
import shutil # Requires python 3.3

def calc_terminal_width():
    res = shutil.get_terminal_size((80, 20))
    return res.columns

class _StatusBuffer:
    def __init__(self, stream):
        self._stream = stream
        self._status_string = ''
        self.lock = Lock()
    def write(self, s):
        try:
            self.lock.acquire()
            self._stream.write('\r%s\r' % (' ' * calc_terminal_width()))
            self._stream.write(s)
#            self._stream.write('\n')
            self._refresh()
        finally:
            self.lock.release()
    def update(self, s):
        try:
            self.lock.acquire()
            self._status_string = s
            self._refresh()
        finally:
            self.lock.release()
    def _refresh(self):
        self._stream.write('\r%s\r' % (' ' * calc_terminal_width()))
        self._stream.write('\033[1;37m%s\033[0m' % self._status_string[:calc_terminal_width()])
        self._stream.flush() # Necessary in python3
    def _atexit(self):
        self._stream.write('\r%s\r' % (' ' * calc_terminal_width()))

class StatusBar:
    threshold = 0
    buffers = {}
    buffers_lock = Lock()
    def __init__(self, stream):
#        self.threshold = 0
        try:
            StatusBar.buffers_lock.acquire()
            if stream in StatusBar.buffers:
                self._out = StatusBar.buffers[stream]
            else:
                self._out = _StatusBuffer(stream)
                StatusBar.buffers[stream] = self._out
            atexit.register(self._out._atexit)
        finally:
            StatusBar.buffers_lock.release()
    def __getitem__(self, verbosity_level):
        def functor(obj):
            if verbosity_level <= StatusBar.threshold:
                self._out.write('%s\n' % obj)
        return functor
    def shutdown(self):
        self.status('')
    def status(self, s):
        '''
        Display the given string as a status line. Note that this does not
        play well with most streams except stderr, and should be used with
        appropriate caution.
        '''
        self._out.update(s)

# By design, python should not run the module defintion more than once
# so the following should not create new instances if loaded twice
MODULE = sys.modules[__name__]
MODULE._singleton_instance = None
MODULE.lock = Lock()

def start_statusbar(stream):
    try:
        MODULE.lock.acquire()
        if MODULE._singleton_instance is None:
            MODULE._singleton_instance = StatusBar(stream)
        else:
            if MODULE._singleton_instance._out._stream != stream:
                MODULE._singleton_instance.shutdown()
                MODULE._singleton_instance = StatusBar(stream)
        return MODULE._singleton_instance
    finally:
        MODULE.lock.release()

def stop_statusbar():
    try:
        MODULE.lock.acquire()
        if MODULE._singleton_instance is not None:
            MODULE._singleton_instance.shutdown()
            MODULE._singelton_instance = None
    finally:
        MODULE.lock.release()
