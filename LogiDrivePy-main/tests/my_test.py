from logidrivepy import LogitechController

# 此文件就是LogiDrivePy-main文件夹下的东西！！直接运行那个就好
controller = LogitechController()

steering_initialize = controller.steering_initialize()
logi_update = controller.logi_update()
is_connected = controller.is_connected(0)

print(f"steering_initialize: {steering_initialize}")
print(f"logi_update: {logi_update}")
print(f"is_connected: {is_connected}")

if steering_initialize and logi_update and is_connected:
    print(f"\n---Logitech Controller Test---")
    while True:
        stateStruct = controller.get_state_engines(0)  # Index 0 corresponds to the first game controller connected
        print('------------------------')
        # for type_ in stateStruct._fields_:
        #     print(type_[0], ': ', type_[1])
        # for i, type_ in enumerate(stateStruct.contents):
        print(stateStruct.contents)
        print(f'state struct lARx axis value: {stateStruct.contents.lARx}')
        print(f'state struct lARy axis value: {stateStruct.contents.lARy}')
        print(f'state struct lAz axis value: {stateStruct.contents.lAZ}')
        print('------------------------')

controller.steering_shutdown()
