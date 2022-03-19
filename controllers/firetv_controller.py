import os
from adb_shell.adb_device import AdbDeviceTcp
from adb_shell.auth.sign_pythonrsa import PythonRSASigner
from adb_shell.auth.keygen import keygen

"""
Copied from:

https://medium.com/codex/controlling-you-fire-tv-with-python-d5e102669066

Key codes:

https://gist.github.com/kibotu/76be44aaa1174bdd252a49a1cd7a02f9
"""

KEYCODES = {
    'select': b'23',
    'back': b'4',
    'up': b'19',
    'down': b'20',
    'left': b'21',
    'right': b'22',
    'play_pause': b'85',
    'rewind': b'89',
    'fast_forward': b'90',
    'menu': b'82',
    'home': b'',
}


class FireTVController():
    def __init__(self):
        if not os.path.isfile('adbkey'):
            print("Generating ADB Keys")
            keygen('adbkey')
        else:
            print('ADB keys found')

        with open('adbkey') as f:
            priv = f.read()

        with open('adbkey.pub') as f:
            pub = f.read()

        self.creds = PythonRSASigner(pub, priv)

    def add_device(self, deviceIP):
        self.device = AdbDeviceTcp(deviceIP, 5555, default_transport_timeout_s=9.)

        try:
            self.device.close()
        except:
            print("No device connected")
        else:
            self.device.connect(rsa_keys=[self.creds], auth_timeout_s=10)
            print("Device Connected")

        return self.device

    def send_command(self, cmd: str):
        assert cmd in KEYCODES.keys()  # ensure that a valid command is passed
        self.device._service(b'shell', b'input keyevent ' + KEYCODES[cmd])


if __name__ == '__main__':

    controller = FireTVController()
    firetv_ip = '192.168.2.138'
    controller.add_device(firetv_ip)

    while True:
        c = input("Give key..")
        if c == 'a':
            controller.send_command('left')
        elif c == 's':
            controller.send_command('down')
        elif c == 'd':
            controller.send_command('right')
        elif c == 'w':
            controller.send_command('up')
        elif c=='e':
            controller.send_command('select')
        elif c=='q':
            controller.send_command('back')