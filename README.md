# 三联屏+罗技驾驶套装+CARLA的demo源码
The source code 包含 carla在三联屏上的显示，罗技G29和carla gym env的双向通信


### 指定显卡
（1）服务端，使用控制台命令开启第一个服务端使carla运行在cuda0：
```
CarlaUE4.exe -quality-level=High -carla-rpc-port=4000 -ini:[/Script/Engine.RendererSettings]:r.GraphicsAdapter=0
```

此时carla使用端口4000，第一个显卡（第二个显卡是2，GraphicsAdapter=1是集成显卡，请勿使用）

（2）客户端
首先指定cuda数据位于哪个显卡：
全部import torch下面加入：
```python
torch.cuda.set_device(1)
```

或
```python
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu") 
```

env中指定连接哪个carla服务端：
```python
self.client = carla.Client("localhost", 4000) 
```

### 三联屏
![image](https://github.com/YoZo-X/logi/logi/figure/screen.png)
```python
        camera_bp_left = self.world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp_middle = self.world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp_right = self.world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp_left.set_attribute("image_size_x", str(1366))
        camera_bp_left.set_attribute("image_size_y", str(768))
        camera_bp_left.set_attribute("fov", str(61))

        camera_bp_middle.set_attribute("image_size_x", str(1366))
        camera_bp_middle.set_attribute("image_size_y", str(768))
        camera_bp_middle.set_attribute("fov", str(55))

        camera_bp_right.set_attribute("image_size_x", str(1366))
        camera_bp_right.set_attribute("image_size_y", str(768))
        camera_bp_right.set_attribute("fov", str(61))

        yaw = 57.5
        x = -5
        z = 3
        pitch = -10
        z_offset = 0.1
        # camera_init_trans = carla.Transform(carla.Location(x=-x, z=z), carla.Rotation(pitch=pitch,yaw=-yaw))
        camera_init_trans = carla.Transform(carla.Location(x=1.2, y=0, z=1.6), carla.Rotation(yaw=-yaw))
        self.camera = self.world.spawn_actor(camera_bp_left, camera_init_trans, attach_to=self.vehicle_ego)

        # camera_init_trans = carla.Transform(carla.Location(x=-x, z=z), carla.Rotation(pitch=-20))
        camera_init_trans = carla.Transform(carla.Location(x=1.2, y=0, z=1.6), carla.Rotation(yaw=+00))
        self.camera1 = self.world.spawn_actor(camera_bp_middle, camera_init_trans, attach_to=self.vehicle_ego)

        # camera_init_trans = carla.Transform(carla.Location(x=-x, z=z), carla.Rotation(pitch=pitch,yaw=yaw))
        camera_init_trans = carla.Transform(carla.Location(x=1.2, y=0, z=1.6 + z_offset), carla.Rotation(yaw=+yaw))
        self.camera2 = self.world.spawn_actor(camera_bp_right, camera_init_trans, attach_to=self.vehicle_ego)

        self.renderObject = self.RenderObject(self.im_width, self.im_height)
        self.renderObject1 = self.RenderObject(self.im_width, self.im_height)
        self.renderObject2 = self.RenderObject(self.im_width, self.im_height)

        self.actor_sensor_list.append(self.camera)
        self.actor_sensor_list.append(self.camera1)
        self.actor_sensor_list.append(self.camera2)

        self.camera.listen(lambda image: self.pygame_callback(image, self.renderObject))
        self.camera1.listen(lambda image: self.pygame_callback(image, self.renderObject1))
        self.camera2.listen(lambda image: self.pygame_callback(image, self.renderObject2))

        pygame.init()
        self.gameDisplay = pygame.display.set_mode((self.im_width, self.im_height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption('Window 1')

        self.gameDisplay1 = pygame.display.set_mode((self.im_width, self.im_height),
                                                    pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption('Window 2')

        # self.gameDisplay2 = pygame.display.set_mode((self.im_width*3, self.im_height), pygame.HWSURFACE | pygame.DOUBLEBUF|pygame.FULLSCREEN)
        self.gameDisplay2 = pygame.display.set_mode((self.im_width * 3, self.im_height),
                                                    pygame.RESIZABLE | pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.NOFRAME | pygame.FULLSCREEN)

        pygame.display.set_caption('Window 3')

        # 填充黑色背景
        self.gameDisplay.fill((0, 0, 0))
        self.gameDisplay1.fill((0, 0, 0))
        self.gameDisplay2.fill((0, 0, 0))

        self.gameDisplay.blit(self.renderObject.surface, (0, 0))
        self.gameDisplay1.blit(self.renderObject1.surface, (0, 0))
        self.gameDisplay2.blit(self.renderObject2.surface, (0, 0))
        for i in range(3):
            if i == 0:
                self.gameDisplay2.blit(self.renderObject.surface, (i * self.im_width, 0))
            if i == 1:
                self.gameDisplay2.blit(self.renderObject1.surface, (i * self.im_width, 0))
            if i == 2:
                self.gameDisplay2.blit(self.renderObject2.surface, (i * self.im_width, 0))

        pygame.display.flip()
```



### 罗技G29和carla env双向通信
  ```python
sys.path.append(r'C:\Users\Estar\Desktop\lda\LogiDrivePy-main\logidrivepy')
sys.path.append(r'C:\Users\Estar\Desktop\lda\LogiDrivePy-main')
from logidrivepy import LogitechController

self.controller = LogitechController()
self.vehicle_ego.apply_control(self.control)
try:
    # pass
    is_wheel_updated = self.controller.logi_update()
    print(f'is_wheel_updated: {is_wheel_updated}')
    print(self.control.steer)
    self.spin_steering_wheel(self.control.steer)
except:
    print('wheel exception')
  ```

### Running
  ```
  python main.py
```

