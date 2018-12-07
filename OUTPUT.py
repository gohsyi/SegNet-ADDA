import time
import sys
from functools import partial
import os
from time import gmtime, strftime


class Output:

    def __init__(self, output_path, note):

        # initialize static print and dynamic print
        self.sprint, self.dprint = init_print()

        # file
        # output_path, note = args.output_path, args.note
        folder = '%s/' % note
        self.output_path = output_path+folder
        self.output_file = self.output_path + 'results.txt'
        self.debug_file = self.output_path + 'debug_log.txt'

        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)

        if os.path.isfile(self.output_path+'.finish'):
            os.remove(self.output_path+'.finish')

        with open(self.output_file, 'w') as _:
            pass

        self.f = open(self.output_file, 'a', 1)
        self.debug_f = open(self.debug_file, 'a', 1)
        self.write(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
        # self.write(str(args))

    def update_restore_path(self, path):
        self.restore_path = path

    def write(self, message):
        # write message to specific file
        self.sprint(message)
        self.f.write(message+'\n')

    def debug_write(self, message, stdout=False):
        # write message to specific file
        if stdout:
            self.sprint(message)
        self.debug_f.write(message+'\n')

    def close(self):
        with open(self.output_path+'.finish', 'w'):
            pass


def init_print(silence=False):
    start_time = time.time()
    sprint = partial(static_print, start_time=start_time)
    dprint = partial(dynamic_print, start_time=start_time, silence=silence)
    return sprint, dprint


def static_print(messages, start_time, silence=False, decorator=None):
    assert type(messages) == str or type(messages) == list
    assert not decorator or decorator == 'both' or decorator == 'before' or decorator == 'after'

    if not silence:
        if type(messages) == str:
            messages = [messages]

        if decorator == 'before' or decorator == 'both':
            print('-'*50)
        for message in messages:
            sys.stdout.write(' ' * 50 + '\r')
            sys.stdout.flush()
            print(message + ' [%is]' % (time.time() - start_time))
        if decorator == 'after' or decorator == 'both':
            print('-'*50)


def dynamic_print(message, start_time, silence=False):
    assert type(message) == str

    if not silence:
        sys.stdout.write(' ' * 110 + '\r')
        sys.stdout.flush()
        sys.stdout.write(message + ' [%is]' % (time.time() - start_time) + '\r')
        sys.stdout.flush()
