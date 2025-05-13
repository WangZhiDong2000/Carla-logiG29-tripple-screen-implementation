import gym
from gym import spaces
import numpy as np
from scipy.integrate import dblquad
from scipy.spatial import distance
from collections import deque
import glob
import os
import sys
from scipy.spatial.distance import cdist
import torch
import carla
import random
import pygame
import numpy as np
import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from skfuzzy import control as ctrl
import math
# from lstm import pre
import time
import torch
import numpy as np
from scipy.integrate import quad

# sys.path.append(r'C:\Users\Estar\Desktop\lda\LogiDrivePy-main\logidrivepy')
# sys.path.append(r'C:\Users\Estar\Desktop\lda\LogiDrivePy-main')
from logidrivepy import LogitechController

start = time.time()

torch.cuda.set_device(1)
try:
    sys.path.append(glob.glob('C:/WindowsNoEditor/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
try:
    sys.path.append(glob.glob('D:/CARLA_0.9.14/WindowsNoEditor/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import math
import random
import carla

state_dim = 7
action_dim = 2
max_action = 1

import gym_carla


class CarlaEnv(gym.Env):
    def __init__(self, port=2000):
        super().__init__()
        self.target_vehicles = {}
        self.consider = 5
        self.im_height = 768
        self.im_width = 1366
        self.action_space = spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), shape=(action_dim,),
                                       dtype=np.float32)
        self.n_target_vehicle = 10
        self.time = 0.1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim * (self.consider + 1),),
                                            dtype=np.float32)
        self.j = 0
        self.i = 0
        self.action = np.zeros([1, action_dim])

        self.state_init = np.zeros(state_dim * (self.consider + 1))
        self.state = self.state_init
        self.world = None
        self.blueprint_library = None
        self.vehicle_ego_bp = None
        self.vehicle_ego = None
        self.camera_bp = None
        self.camera = None
        self.current_map = None
        self.lane_id = None
        self.spectator = None
        self.col_sensor = None
        self.collision_with_sur = False
        # self.fail_counter = 0
        self.uzenet = None
        self.vehicle_ego = None
        self.initial_distance = 0
        self.initial_vehicle_speed = 0
        self.initial_speed = 0
        self.sensor_queue = None
        self.actor_vehicle_list = []
        self.actor_sensor_list = []
        self.client = carla.Client('localhost', port=port)
        self.client.set_timeout(20.0)
        self.client.load_world('/Game/Carla/Maps/Town06')
        self.world = self.client.get_world()
        self.current_map = self.world.get_map()
        self.blueprint_library = self.world.get_blueprint_library()
        self.original_settings = self.world.get_settings()
        self.settings = self.world.get_settings()
        self.settings.fixed_delta_seconds = self.time
        self.settings.synchronous_mode = True
        self.world.apply_settings(self.settings)
        random.seed(0)

        self.lon_e_buffer = deque(maxlen=800)
        self.lat_e_buffer = deque(maxlen=800)

        self.random = 3.5 * np.random.uniform(0, 1)

        self.transform_ego = carla.Transform(carla.Location(x=-252.2, y=247.2 - self.random, z=0.50000),
                                             carla.Rotation(pitch=0.000000, yaw=0, roll=0.000000))

        self.transform_ego_destination = carla.Transform(carla.Location(x=-205.2, y=247.2, z=0.50000)).location
        self.transform_dict = {}
        self.transform_destination_dict = {}

        n = random.random()
        self.transform_dict[f"transform{1}"] = carla.Transform(
            carla.Location(-252.2, 247.2 + 3.5, 0.50000),
            carla.Rotation(pitch=0.000000, yaw=0, roll=0.000000))
        self.transform_dict[f"transform{2}"] = carla.Transform(
            carla.Location(-252.2 + 11, 247.2 + 3.5, 0.50000),
            carla.Rotation(pitch=0.000000, yaw=0, roll=0.000000))
        self.transform_dict[f"transform{3}"] = carla.Transform(
            carla.Location(-252.2 + 14, 247.2 - 3.5, 0.50000),
            carla.Rotation(pitch=0.000000, yaw=0, roll=0.000000))
        self.transform_dict[f"transform{4}"] = carla.Transform(
            carla.Location(-252.2 + 21, 247.2 - 3.5 * n / (3 - 1 + n), 0.50000),
            carla.Rotation(pitch=0.000000, yaw=0, roll=0.000000))
        self.transform_dict[f"transform{5}"] = carla.Transform(
            carla.Location(-252.2 + 30, 247.2 - 3.5 * n / (4 - 1 + n), 0.50000),
            carla.Rotation(pitch=0.000000, yaw=0, roll=0.000000))
        self.transform_dict[f"transform{6}"] = carla.Transform(
            carla.Location(-252.2 + 40, 247.2 - 3.5, 0.50000),
            carla.Rotation(pitch=0.000000, yaw=0, roll=0.000000))
        self.transform_dict[f"transform{7}"] = carla.Transform(
            carla.Location(-252.2 + 49, 247.2 - 3.5, 0.50000),
            carla.Rotation(pitch=0.000000, yaw=0, roll=0.000000))
        self.transform_dict[f"transform{8}"] = carla.Transform(
            carla.Location(-252.2 + 50, 247.2 + 3.5, 0.50000),
            carla.Rotation(pitch=0.000000, yaw=0, roll=0.000000))
        self.transform_dict[f"transform{9}"] = carla.Transform(
            carla.Location(-252.2 + 60, 247.2 + 3.5 * n / (3 - 1 + n), 0.50000),
            carla.Rotation(pitch=0.000000, yaw=0, roll=0.000000))
        self.transform_dict[f"transform{10}"] = carla.Transform(
            carla.Location(-252.2 + 70, 247.2 + 3.5 * n / (4 - 1 + n), 0.50000),
            carla.Rotation(pitch=0.000000, yaw=0, roll=0.000000))
        self.sb = True
        self.sb1 = True

        self.s = 7  # 跟车距离
        self.lane_width = self.current_map.get_waypoint(self.transform_ego.location).lane_width

        self.x_0_EV_glo = np.array([self.transform_ego.location.x, 0 / 3.6, 0, self.transform_ego.location.y, 0,
                                    0])  # initial state of EV in global frame
        self.x_0_EV_loc = np.array([self.transform_ego.location.x, self.transform_ego.location.y, 0, 0 / 3.6, 0, 0, 0,
                                    0])  # initial state of EV in vehicle frame
        self.X_State_0 = [self.x_0_EV_glo]
        for i in range(self.n_target_vehicle):
            transform_key = f'transform{i + 1}'
            transform_location = self.transform_dict[transform_key].location
            initial_state = np.array([transform_location.x, 0, 0, transform_location.y, 0, 0])
            self.X_State_0.append(initial_state)
        self.X_State = list()
        self.X_State.append(self.X_State_0)
        self.index_EV = 0
        self.MU = list()  # the probability
        self.P = list()  # the covariance matrix in IMM-KF
        self.Y = list()  # the measurement
        self.M = list()  # the activated model -- models which are effective
        self.X_Pre = list()  # the prediction
        self.X_Po_All = list()  # the all possible predictions
        self.Ref_Speed = list()  # reference speed of cars
        self.X_Hat = list()
        self.num = 1

        self.speed = []
        self.average_speed = 0
        self.acclerationy = []
        self.average_acclerationy = 0
        self.collision = 0
        self.jerk = []
        self.average_jerk = 0
        self.jilusudu = 0
        self.lane_range = [247.2 + 1.5 * 3.5, 247.2 - 1.5 * 3.5]
        self.reward4 = np.zeros(self.n_target_vehicle + 1)
        self.reward5 = 0
        self.L_Center = [250.7, 247.2, 243.7]  # lane center positions
        self.ob1 = 0
        self.control = carla.VehicleControl()
        self.index = 0
        self.juli = []
        self.next = []

        # steering wheel
        self.controller = LogitechController()
        self.controller.steering_initialize()


    def render(self, mode='human'):
        pass

    def set_synchronous_mode(self, synchronous=True):
        """Set whether to use the synchronous mode"""
        self.settings.synchronous_mode = synchronous
        self.world.apply_settings(self.settings)

    def destroy(self):
        for actor in self.actor_vehicle_list:
            actor.destroy()
        for actor in self.actor_sensor_list:
            actor.destroy()

    def close(self):
        """Reverts back to original settings"""
        self.world.apply_settings(self.original_settings)
        self.set_synchronous_mode(False)

    def collevent(self, event):

        eventactor = event.actor
        eventotheractor = event.other_actor
        if eventotheractor in self.target_vehicles.values():
            print("COLLISION!!!!!!!!!!!")
            self.collision_with_sur = True
        if eventactor.id == self.vehicle_ego.id:
            print("COLLISION!!!!!!!!!!!")
            self.collision_with_sur = True

    def reset(self):
        """Resets CARLA environment"""
        self.target_vehicles = {}
        # self.controller = LogitechController()
        # self.controller.steering_initialize()

        n = random.random()
        self.transform_dict[f"transform{1}"] = carla.Transform(
            carla.Location(-252.2, 247.2 + 3.5, 0.50000),
            carla.Rotation(pitch=0.000000, yaw=0, roll=0.000000))
        self.transform_dict[f"transform{2}"] = carla.Transform(
            carla.Location(-252.2 + 11, 247.2 + 3.5, 0.50000),
            carla.Rotation(pitch=0.000000, yaw=0, roll=0.000000))
        self.transform_dict[f"transform{3}"] = carla.Transform(
            carla.Location(-252.2 + 14, 247.2 - 3.5, 0.50000),
            carla.Rotation(pitch=0.000000, yaw=0, roll=0.000000))
        self.transform_dict[f"transform{4}"] = carla.Transform(
            carla.Location(-252.2 + 22, 247.2 - 3.5 * n / (3 - 1 + n), 0.50000),
            carla.Rotation(pitch=0.000000, yaw=0, roll=0.000000))
        self.transform_dict[f"transform{5}"] = carla.Transform(
            carla.Location(-252.2 + 30, 247.2 - 3.5 * n / (4 - 1 + n), 0.50000),
            carla.Rotation(pitch=0.000000, yaw=0, roll=0.000000))
        self.transform_dict[f"transform{6}"] = carla.Transform(
            carla.Location(-252.2 + 40, 247.2 - 3.5, 0.50000),
            carla.Rotation(pitch=0.000000, yaw=0, roll=0.000000))
        self.transform_dict[f"transform{7}"] = carla.Transform(
            carla.Location(-252.2 + 49, 247.2 - 3.5, 0.50000),
            carla.Rotation(pitch=0.000000, yaw=0, roll=0.000000))
        self.transform_dict[f"transform{8}"] = carla.Transform(
            carla.Location(-252.2 + 52, 247.2 + 3.5, 0.50000),
            carla.Rotation(pitch=0.000000, yaw=0, roll=0.000000))
        self.transform_dict[f"transform{9}"] = carla.Transform(
            carla.Location(-252.2 + 60, 247.2 + 3.5 * n / (3 - 1 + n), 0.50000),
            carla.Rotation(pitch=0.000000, yaw=0, roll=0.000000))
        self.transform_dict[f"transform{10}"] = carla.Transform(
            carla.Location(-252.2 + 70, 247.2 + 3.5 * n / (4 - 1 + n), 0.50000),
            carla.Rotation(pitch=0.000000, yaw=0, roll=0.000000))
        # self.transform_ego = carla.Transform(carla.Location(x=-252.2+40, y=247.2 , z=0.50000),
        #                                      carla.Rotation(pitch=0.000000, yaw=0, roll=0.000000))

        self.j = 0
        self.lon_e_buffer.clear()
        self.lat_e_buffer.clear()

        try:
            if len(self.actor_vehicle_list) > 0:
                self.destroy()
                self.actor_vehicle_list = []
            if len(self.actor_sensor_list) > 0:
                self.destroy()
                self.actor_sensor_list = []
        except:
            self.actor_sensor_list = []
            self.actor_vehicle_list = []
        self.random = 3.5 * np.random.uniform(0, 1)
        self.transform_ego = carla.Transform(carla.Location(x=-252.2, y=247.2 - self.random, z=0.50000),
                                             carla.Rotation(pitch=0.000000, yaw=0, roll=0.000000))
        self.transform_ego_destination = carla.Transform(carla.Location(x=-205.2, y=247.2, z=0.50000)).location

        self.setup_target_vehicle()
        self.setup_vehicle_ego()

        self.action = np.zeros([1, action_dim])
        self.state = self.state_init

        self.sb = True
        self.sb1 = True

        self.x_0_EV_glo = np.array([self.transform_ego.location.x, 0 / 3.6, 0, self.transform_ego.location.y, 0,
                                    0])  # initial state of EV in global frame
        self.x_0_EV_loc = np.array([self.transform_ego.location.x, self.transform_ego.location.y, 0, 0 / 3.6, 0, 0, 0,
                                    0])  # initial state of EV in vehicle frame
        self.X_State_0 = [self.x_0_EV_glo]
        for i in range(self.n_target_vehicle):
            transform_key = f'transform{i + 1}'
            transform_location = self.transform_dict[transform_key].location
            initial_state = np.array([transform_location.x, 0, 0, transform_location.y, 0, 0])
            self.X_State_0.append(initial_state)
        self.X_State = list()
        self.X_State.append(self.X_State_0)
        self.index_EV = 0
        self.MU = list()  # the probability
        self.P = list()  # the covariance matrix in IMM-KF
        self.Y = list()  # the measurement
        self.M = list()  # the activated model -- models which are effective
        self.X_Pre = list()  # the prediction
        self.X_Po_All = list()  # the all possible predictions
        self.Ref_Speed = list()  # reference speed of cars
        self.X_Hat = list()
        self.num = 1

        self.speed = []
        self.average_speed = 0
        self.acclerationy = []
        self.average_acclerationy = 0
        self.jerk = []
        self.average_jerk = 0
        self.jilusudu = 0
        self.lane_range = [247.2 + 1.5 * 3.5, 247.2 - 1.5 * 3.5]
        self.reward4 = np.zeros(self.n_target_vehicle + 1)
        self.reward5 = 0
        self.ob1 = 0
        self.index = 0
        self.juli = []
        self.reward4 = np.zeros(self.n_target_vehicle + 1)
        self.reward5 = 0
        self.ob1 = 0
        self.control = carla.VehicleControl()
        self.index = 0
        self.juli = []

        return self.state, False

    def spect_cam(self, vehicle):
        self.spectator = self.world.get_spectator()
        # tr = vehicle.get_transform()
        tr = carla.Transform(carla.Location(x=-252.2, y=247.2 - self.random, z=0.50000),
                             carla.Rotation(pitch=0.000000, yaw=0, roll=0.000000))
        tr.location.z += 80
        tr.location.x += 45
        wp = self.current_map.get_waypoint(vehicle.get_transform().location, project_to_road=True,
                                           lane_type=carla.LaneType.Driving)
        tr.rotation = carla.Rotation(pitch=-90.000000, yaw=-0.289116, roll=0.000000)
        self.spectator.set_transform(tr)

    def setup_vehicle_ego(self):
        self.collision_with_sur = False
        ped_suc = False
        while ped_suc == False:
            try:
                ped_suc = True
                self.vehicle_ego_bp = self.blueprint_library.find('vehicle.audi.etron')
                self.vehicle_ego_bp.set_attribute('role_name', 'hero')


                self.vehicle_ego = self.world.spawn_actor(self.vehicle_ego_bp, self.transform_ego)
                self.actor_vehicle_list.append(self.vehicle_ego)

            except:
                print('FAILED TO MOVE vehicle_ego')
                ped_suc = False
        col_bp = self.world.get_blueprint_library().find('sensor.other.collision')
        self.col_sensor = self.world.spawn_actor(col_bp, carla.Transform(), attach_to=self.vehicle_ego)
        self.col_sensor.listen(lambda event: self.collevent(event))
        self.vehicle_ego.set_autopilot(True)
        self.actor_sensor_list.append(self.col_sensor)
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

        self.lane_width = self.current_map.get_waypoint(self.transform_ego.location).lane_width
        self.lane_id = self.current_map.get_waypoint(self.transform_ego.location, project_to_road=True,
                                                     lane_type=carla.LaneType.Driving).lane_id
        self.initial_vehicle_speed = 6 + random.random() * 5
        self.uzenet = 'AVOIDED PEDESTRIAN'

    def setup_target_vehicle(self):  # GYALOGOS LÉTREHOZÁSA
        self.initial_distance = random.random() * 10 + 8
        self.initial_speed = min(6, (
                    0.5 + 4 * random.random()) * self.initial_vehicle_speed / self.initial_distance)  # random.random()*2+0.5

        for i in range(1, self.n_target_vehicle + 1):
            ped_suc = False
            while ped_suc == False:
                try:
                    ped_suc = True
                    target_vehicle = self.world.spawn_actor(self.blueprint_library.filter("model3")[0],
                                                            self.transform_dict[f'transform{i}'])
                    self.target_vehicles[f"target_vehicle_{i}"] = target_vehicle  # 将生成的目标车辆存储在字典中
                    self.actor_vehicle_list.append(target_vehicle)  # 将生成的目标车辆添加到列表中
                except IndexError:
                    print("IndexError: No blueprint found for 'model3'.")
                    ped_suc = False
                except Exception as e:
                    print(f"Failed to move vehicle_{i} due to exception: {e}")
                    ped_suc = False

    def state_get(self, world_snapshot, currentmap, vehicle, sur):
        vehicle_snapshot = world_snapshot.find(vehicle.id)
        sur_snapshot = world_snapshot.find(sur.id)
        vehicle_transform = vehicle_snapshot.get_transform()
        vehicle_velocity = vehicle_snapshot.get_velocity()
        sur_transform = sur_snapshot.get_transform()
        sur_velocity = sur_snapshot.get_velocity()
        p_veh = (vehicle_transform.location.x + 4.79 / 2 * math.cos(vehicle_transform.rotation.yaw),
                 vehicle_transform.location.y + 4.79 / 2 * math.sin(vehicle_transform.rotation.yaw))
        p_sur = (sur_transform.location.x + 4.79 / 2 * math.cos(sur_transform.rotation.yaw),
                 sur_transform.location.y + 4.79 / 2 * math.sin(sur_transform.rotation.yaw))
        vector1 = np.array(
            [math.cos(vehicle_transform.rotation.yaw * math.pi / 180),
             math.sin(vehicle_transform.rotation.yaw * math.pi / 180)])
        vector2 = np.subtract(p_sur, p_veh)
        dotproduct = np.dot(vector1, vector2)

        angle = np.arccos(
            np.sign(dotproduct) * dotproduct / math.sqrt(vector2[0] * vector2[0] + vector2[1] * vector2[1]))
        lane_offset_x = 0
        lane_offset_y = 0
        temp_loc = carla.Location(sur_transform.location.x + lane_offset_x,
                                  sur_transform.location.y + lane_offset_y, sur_transform.location.z)
        road_waypoint = currentmap.get_waypoint(temp_loc, project_to_road=True, lane_type=carla.LaneType.Driving)

        p_veh = (sur_transform.location.x + 4.79 / 2 * math.cos(sur_transform.rotation.yaw),
                 sur_transform.location.y + 4.79 / 2 * math.sin(sur_transform.rotation.yaw))
        p_road = (road_waypoint.transform.location.x, road_waypoint.transform.location.y)
        v_veh_road = (p_veh[0] - p_road[0], p_veh[1] - p_road[1])
        v_road = (math.cos(road_waypoint.transform.rotation.yaw * math.pi / 180),
                  math.sin(road_waypoint.transform.rotation.yaw * math.pi / 180))
        lat_err = -distance.euclidean(p_veh, p_road) * np.sign(np.cross(v_veh_road, v_road))
        ang_err = sur_transform.rotation.yaw - road_waypoint.transform.rotation.yaw
        if ang_err > 180:
            ang_err = -360 + ang_err
        if ang_err < -180:
            ang_err = 360 + ang_err

        return vector2[0], vector2[1] / (self.lane_range[0] - self.lane_range[
            1]), angle, vehicle_velocity.x - sur_velocity.x, vehicle_velocity.y - sur_velocity.y, v_veh_road[1], ang_err

    def sur_sensor(self, vehicle_tf, sur_tf):  # PEDESTRIAN DISTANCE AND ANGLE

        p_veh = (vehicle_tf.location.x + 2.5 * math.cos(vehicle_tf.rotation.yaw),
                 vehicle_tf.location.y + 2.5 * math.sin(vehicle_tf.rotation.yaw))
        p_sur = (sur_tf.location.x, sur_tf.location.y)
        d_sur_veh = distance.euclidean(p_veh, p_sur)
        vector1 = np.array(
            [math.cos(vehicle_tf.rotation.yaw * math.pi / 180), math.sin(vehicle_tf.rotation.yaw * math.pi / 180)])
        vector2 = np.subtract(p_sur, p_veh)
        dotproduct = np.dot(vector1, vector2)

        angle = np.arccos(
            np.sign(dotproduct) * dotproduct / math.sqrt(vector2[0] * vector2[0] + vector2[1] * vector2[1]))
        d_sur_veh_reciprok = 0
        if d_sur_veh == 0:
            d_sur_veh = 0.001

        d_sur_veh_reciprok = 1 / d_sur_veh
        # self.fail_counter = 0
        return d_sur_veh_reciprok, angle

    def snap_to_s_t(self, world_snapshot, currentmap, vehicle, sur):  # OB FÜGGVÉNY
        vehicle_snapshot = world_snapshot.find(vehicle.id)
        sur_snapshot = world_snapshot.find(sur.id)
        vehicle_transform = vehicle_snapshot.get_transform()
        vehicle_velocity = vehicle_snapshot.get_velocity()
        vehicle_ang_vel = vehicle_snapshot.get_angular_velocity()
        vehicle_acceleration = vehicle_snapshot.get_acceleration()

        sur_transform = sur_snapshot.get_transform()

        d_sur_veh, angle_sur_veh = self.sur_sensor(vehicle_transform, sur_transform)

        lane_offset_x = 0
        lane_offset_y = 0
        i = 1
        while True:
            temp_loc = carla.Location(vehicle_transform.location.x + lane_offset_x,
                                      vehicle_transform.location.y + lane_offset_y, vehicle_transform.location.z)
            road_waypoint = currentmap.get_waypoint(temp_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
            l_id = road_waypoint.lane_id
            if l_id == self.lane_id:
                break
            # lane_offset_x = np.power(-1, i)*i/2#*#math.cos(road_waypoint_old.transform.rotation.yaw*math.pi/180)
            lane_offset_y = np.power(-1, i) * i  # *math.sin(road_waypoint_old.transform.rotation.yaw*math.pi/180)

            i += 1
            if i == 100:
                print("i>100")
            if i == 200:
                break

        p_veh = (vehicle_transform.location.x, vehicle_transform.location.y)
        p_road = (road_waypoint.transform.location.x, road_waypoint.transform.location.y)

        v_veh_road = (p_veh[0] - p_road[0], p_veh[1] - p_road[1])
        v_road = (math.cos(road_waypoint.transform.rotation.yaw * math.pi / 180),
                  math.sin(road_waypoint.transform.rotation.yaw * math.pi / 180))
        lat_err = -distance.euclidean(p_veh, p_road) * np.sign(np.cross(v_veh_road, v_road))

        ang_err = vehicle_transform.rotation.yaw - road_waypoint.transform.rotation.yaw
        if ang_err > 180:
            ang_err = -360 + ang_err
        if ang_err < -180:
            ang_err = 360 + ang_err
        speed = math.sqrt(vehicle_velocity.x ** 2 + vehicle_velocity.y ** 2)
        acc = math.sqrt(vehicle_acceleration.x ** 2 + vehicle_acceleration.y ** 2)
        speedX = abs(speed * math.cos(
            vehicle_transform.rotation.yaw * math.pi / 180 - math.atan2(vehicle_velocity.y, vehicle_velocity.x)))
        speedY = speed * math.sin(
            vehicle_transform.rotation.yaw * math.pi / 180 - math.atan2(vehicle_velocity.y, vehicle_velocity.x))
        ang_vel = vehicle_ang_vel.z
        accX = acc * math.cos(
            vehicle_transform.rotation.yaw * math.pi / 180 - math.atan2(vehicle_acceleration.y, vehicle_acceleration.x))
        accY = acc * math.sin(
            vehicle_transform.rotation.yaw * math.pi / 180 - math.atan2(vehicle_acceleration.y, vehicle_acceleration.x))
        return np.hstack((lat_err, ang_err, speedX, speedY, ang_vel, accX, accY, d_sur_veh, angle_sur_veh))

    def state_ego_get(self, world_snapshot, currentmap, theta):
        vehicle_snapshot = world_snapshot.find(self.vehicle_ego.id)
        vehicle_transform = vehicle_snapshot.get_transform()
        vehicle_velocity = vehicle_snapshot.get_velocity()
        vehicle_acceleration = vehicle_snapshot.get_acceleration()
        lane_offset_x = 0
        lane_offset_y = 0
        temp_loc = carla.Location(vehicle_transform.location.x + lane_offset_x,
                                  vehicle_transform.location.y + lane_offset_y, vehicle_transform.location.z)
        road_waypoint = currentmap.get_waypoint(temp_loc, project_to_road=True, lane_type=carla.LaneType.Driving)

        p_veh = (vehicle_transform.location.x, vehicle_transform.location.y)
        p_road = (road_waypoint.transform.location.x, road_waypoint.transform.location.y)

        v_veh_road = (p_veh[0] - p_road[0], p_veh[1] - p_road[1])
        v_road = (math.cos(road_waypoint.transform.rotation.yaw * math.pi / 180),
                  math.sin(road_waypoint.transform.rotation.yaw * math.pi / 180))
        lat_err = -distance.euclidean(p_veh, p_road) * np.sign(np.cross(v_veh_road, v_road))

        ang_err = vehicle_transform.rotation.yaw - road_waypoint.transform.rotation.yaw
        if ang_err > 180:
            ang_err = -360 + ang_err
        if ang_err < -180:
            ang_err = 360 + ang_err
        speed = math.sqrt(vehicle_velocity.x ** 2 + vehicle_velocity.y ** 2)
        acc = math.sqrt(vehicle_acceleration.x ** 2 + vehicle_acceleration.y ** 2)
        speedX = abs(speed * math.cos(
            vehicle_transform.rotation.yaw * math.pi / 180 - math.atan2(vehicle_velocity.y, vehicle_velocity.x)))
        speedY = speed * math.sin(
            vehicle_transform.rotation.yaw * math.pi / 180 - math.atan2(vehicle_velocity.y, vehicle_velocity.x))
        accX = acc * math.cos(
            vehicle_transform.rotation.yaw * math.pi / 180 - math.atan2(vehicle_acceleration.y, vehicle_acceleration.x))
        accY = acc * math.sin(
            vehicle_transform.rotation.yaw * math.pi / 180 - math.atan2(vehicle_acceleration.y, vehicle_acceleration.x))
        return (self.lane_range[0] - vehicle_transform.location.y) / (self.lane_range[0] - self.lane_range[
            1]), vehicle_transform.rotation.yaw * math.pi / 180, theta, speedX, accX, v_veh_road[1], ang_err

    def jilu(self, writer):

        writer.add_scalar("average_speed", self.average_speed, self.i)
        writer.add_scalar("average_acclerationy", self.average_acclerationy, self.i)
        writer.add_scalar("average_collision", self.collision / self.i * 100, self.i)
        writer.add_scalar("average_jerk", self.average_jerk / self.settings.fixed_delta_seconds, self.i)

    def step(self, u):
        # for actor in self.world.get_actors():
        #     if actor.attributes.get("role_name") in ["hero", "ego_vehicle"]:
        #         print(actor)
        self.world.tick()
        self.j += 1
        at = u
        done = False
        self.spect_cam(self.vehicle_ego)
        world_snapshot = self.world.get_snapshot()
        control = carla.VehicleControl()

        control.steer = 0
        control.throttle = 0.2
        control.brake = 0

        if self.j % 3 == 0:
            self.control = control
        if self.control.throttle > 0:
            action = [self.control.steer, self.control.throttle]
        else:
            action = [self.control.steer, self.control.brake]

        self.transform_destination_dict[f"transform{3 + 1}"] = carla.Transform(
            carla.Location(-252.2 + 100, 247.2 + 3.5, 0.50000),
            carla.Rotation(pitch=0.000000, yaw=0, roll=0.000000)).location
        self.transform_dict[f"transform{111}"] = carla.Transform(
            carla.Location(-200, 247.2 + 3.5, 0.50000),
            carla.Rotation(pitch=0.000000, yaw=0, roll=0.000000))

        if world_snapshot.find(
                self.target_vehicles[f"target_vehicle_{6}"].id).get_transform().location.x > world_snapshot.find(
            self.vehicle_ego.id).get_transform().location.x > -252.2 + 21:
            self.vehicle_ego.apply_control(self.control)

        else:
            self.vehicle_ego.apply_control(self.control)
        try:
            # is_wheel_updated = self.controller.logi_update()
            # print(f'is_wheel_updated: {is_wheel_updated}')
            force = int(random.random() * 50)
            print(f'force: {force}')
            self.controller.LogiPlaySpringForce(0, force, 100, 40)
            self.controller.logi_update()

        except:
            print('wheel exception')

        for i in range(1, self.n_target_vehicle + 1):
            self.target_vehicles[f"target_vehicle_{i}"].apply_control(control)

        State = []
        ego_state = self.state_ego_get(world_snapshot, self.current_map, float(at[0]))

        for sur in self.actor_vehicle_list:
            if sur.id == self.vehicle_ego.id:
                pass
            else:
                state = self.state_get(world_snapshot, self.current_map, self.vehicle_ego, sur)
                State.append(state)
        State.sort(key=lambda x: np.abs(x[0]))
        State = State[:self.consider]

        State.insert(0, ego_state)

        State = [item for sublist in State for item in sublist]
        vehicle_snapshot = world_snapshot.find(self.vehicle_ego.id)
        vehicle_acceleration = vehicle_snapshot.get_acceleration()
        vehicle_transform = vehicle_snapshot.get_transform()

        # acc = math.sqrt(vehicle_acceleration.x ** 2 + vehicle_acceleration.y ** 2)
        koztes_cucc = np.abs((np.abs(ego_state[6]) + np.abs(self.ob1 - ego_state[6]) * 3) / 180 * math.pi) * 3

        reward1 = np.cos(math.sqrt(np.abs(koztes_cucc * math.pi * 2 / 3))) / 4 - 0.25
        reward2 = 0.2 * vehicle_acceleration.x + 0.6 * np.exp(
            -0.25 * np.abs(self.initial_vehicle_speed - np.sqrt(ego_state[3] ** 2)))
        reward2 = reward2.clip(-0.3, 0.6)
        reward3 = 0.5 * np.exp(-0.4 * np.abs(vehicle_transform.location.y - self.L_Center[0])) + 0.5 * np.exp(
            -0.4 * np.abs(vehicle_transform.location.y - self.L_Center[1])) + 0.5 * np.exp(
            -0.4 * np.abs(vehicle_transform.location.y - self.L_Center[2])) - 0.555  # lane center positions
        reward = reward1 + reward2 + reward3 + self.reward5
        # print(reward, reward1, reward2, reward3, self.reward5)
        if world_snapshot.find(self.vehicle_ego.id).get_transform().location.x < world_snapshot.find(
                self.target_vehicles[f"target_vehicle_{4}"].id).get_transform().location.x:
            self.index = 0
        elif world_snapshot.find(
                self.target_vehicles[f"target_vehicle_{9}"].id).get_transform().location.x > world_snapshot.find(
            self.vehicle_ego.id).get_transform().location.x > world_snapshot.find(
            self.target_vehicles[f"target_vehicle_{6}"].id).get_transform().location.x:
            self.index = 1
        else:
            self.index = 2
        juli = [world_snapshot.find(self.vehicle_ego.id).get_transform().location.x,
                world_snapshot.find(self.vehicle_ego.id).get_transform().location.y,
                world_snapshot.find(self.vehicle_ego.id).get_transform().rotation.yaw * math.pi / 180]
        # print(juli)
        self.juli.append(juli)

        # self.uzenet = '!!!STOPPED!!!'
        if abs(ego_state[6]) > 60:
            done = True
            reward += -5
        if world_snapshot.find(self.vehicle_ego.id).get_transform().location.y < self.lane_range[
            1] or world_snapshot.find(self.vehicle_ego.id).get_transform().location.y > self.lane_range[0]:
            # if abs(ob[1]) > 90:
            done = True
            reward += -5
            self.collision += 1
        if self.collision_with_sur:
            done = True
            self.collision += 1
            self.uzenet = '!!!COLLISION!!!'
            reward = -10
        if world_snapshot.find(self.vehicle_ego.id).get_transform().location.x > -252.2 + 70:
            done = True
            reward += 10 - np.abs(ego_state[6]) / 180 * math.pi * 10
            reward = 5 if reward < 5 else reward
        if done:
            self.i += 1
            reward += (world_snapshot.find(
                self.vehicle_ego.id).get_transform().location.x - self.transform_ego.location.x) / 10
            np.save('vehicle_location.npy', self.juli)
            juli = []
            for i in range(1, self.n_target_vehicle + 1):
                juli1 = [world_snapshot.find(self.target_vehicles[f"target_vehicle_{i}"].id).get_transform().location.x,
                         world_snapshot.find(self.target_vehicles[f"target_vehicle_{i}"].id).get_transform().location.y,
                         world_snapshot.find(self.target_vehicles[
                                                 f"target_vehicle_{i}"].id).get_transform().rotation.yaw * math.pi / 180]
                juli.append(juli1)
            np.save('vehicle_location1.npy', juli)

        # self.gameDisplay.blit(self.renderObject.surface, (0, 0))
        # self.gameDisplay1.blit(self.renderObject1.surface, (0, 0))
        # self.gameDisplay2.blit(self.renderObject2.surface, (0, 0))
        for i in range(3):
            if i == 0:
                self.gameDisplay2.blit(self.renderObject.surface, (i * self.im_width, 0))
            if i == 1:
                self.gameDisplay2.blit(self.renderObject1.surface, (i * self.im_width, 0))
            if i == 2:
                self.gameDisplay2.blit(self.renderObject2.surface, (i * self.im_width, 0))

        pygame.display.flip()
        # 获取pygame事件
        for event in pygame.event.get():
            # If the window is closed, break the while loop
            if event.type == pygame.QUIT:
                crashed = True
        vehicle_snapshot = world_snapshot.find(self.vehicle_ego.id)
        vehicle_velocity = vehicle_snapshot.get_velocity()
        vehicle_acceleration = vehicle_snapshot.get_acceleration()
        self.speed.append(np.sqrt(vehicle_velocity.x ** 2 + (vehicle_velocity.y ** 2)))
        self.average_speed = np.mean(self.speed)
        self.acclerationy.append(np.abs(vehicle_acceleration.y))
        self.average_acclerationy = np.mean(self.acclerationy)
        self.jerk.append(vehicle_acceleration.x - self.jilusudu)
        self.jilusudu = vehicle_acceleration.x
        self.average_jerk = np.mean(self.jerk)
        self.ob1 = ego_state[6]
        dic = {}
        dic["action"] = action
        dic["act"] = self.index
        return State, reward, np.array([done], dtype=np.bool_), dic

    class RenderObject(object):
        def __init__(self, width, height):
            init_image = np.random.randint(0, 255, (height, width, 3), dtype='uint8')
            self.surface = pygame.surfarray.make_surface(init_image.swapaxes(0, 1))

    def pygame_callback(self, data, obj):
        img = np.reshape(np.copy(data.raw_data), (data.height, data.width, 4))
        img = img[:, :, :3]
        img = img[:, :, ::-1]
        obj.surface = pygame.surfarray.make_surface(img.swapaxes(0, 1))
