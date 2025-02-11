import sys
sys.path.append('../logidrivepy')
from logidrivepy import LogitechController
import time
import ctypes
import tkinter as tk

def spin_controller(controller, test_spin=False):
    print(f'is wheel connected: {controller.LogiIsConnected(0)}')
    if not test_spin:
        for i in range(-100, 0, 2):  # for i in range(-100, 102, 2):
            controller.LogiPlaySpringForce(0, i, 100, 40)
            # controller.play_spring_force(0, i, 100, 40)  # DO NOT RUN THIS LINE
            controller.logi_update()
            time.sleep(0.1)
    else:
        # for i in [0, 0, -6, 6]:  # first value seems to be omitted, spin to a specified angle e.g 0
        #     r = controller.LogiPlaySpringForce(0, i, 100, 40)
        #     print(f'return: {r}')
        #     controller.logi_update()
        #     time.sleep(0.1)

        r = False
        while not r:  # first value seems to be omitted, spin to a specified angle e.g 0
            con = controller.LogiIsDeviceConnected(0, 0)
            print(f'con: {con}')
            r = controller.LogiPlaySpringForce(0, 6, 100, 40)
            print(f'return: {r}')
            controller.logi_update()
            time.sleep(0.1)

def spin_test(test_spin=False):
    controller = LogitechController()

    controller.steering_initialize()
    print("\n---Logitech Spin Test---")
    # spin_controller(controller, test_spin)
    # controller.logi_update()
    time.sleep(5.0)

    r = controller.LogiPlaySpringForce(0, -50, 100, 40)
    print(r)
    # controller.logi_update()
    print("Spin test passed.\n")

    print('sleep 5 secs')
    time.sleep(5)

    # print('exit')
    # exit()
    controller.steering_shutdown()


if __name__ == "__main__":
    spin_test(test_spin=True)
