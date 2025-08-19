import glob
import os
import sys
import random
import time
import numpy as np
import cv2
import math
from settings import *

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
carla_root = r"D:\CARLA0914\WindowsNoEditor\PythonAPI"
sys.path.append(carla_root)
sys.path.append(os.path.join(carla_root, "carla"))
import carla
from agents.navigation.global_route_planner import GlobalRoutePlanner




# Class for the carla environment
class CarEnv:
    SHOW_CAM = SHOW_PREVIEW
    STEER_AMT = 1.0
    im_height = IM_HEIGHT
    im_width = IM_WIDTH
    front_camera = None

    def __init__(self):
        self. client = carla.Client('127.0.0.1', 2000)
        self.client.set_timeout(5)
        self.world = self.client.get_world()
        settings = self.world.get_settings()
        if not settings.synchronous_mode:
            settings.synchronous_mode = True          # 开启同步
            settings.fixed_delta_seconds = 1.0 / FPS  # 与你的 FPS 对齐
            self.world.apply_settings(settings)
        self.blueprint_library = self.world.get_blueprint_library()
        self.prius = self.blueprint_library.filter('prius')[0]
        self.actor_list   = []          # 存放本轮创建的所有 actor
        self.collision_hist = []
        self.location_hist  = []
        self.dist_traveled  = 0

    def reset(self):
    # ========= 0 先清理上一次 =========
        for actor in self.actor_list:
            self._safe_destroy(actor)
        self.actor_list.clear()        # 一次就够
        self.collision_hist.clear()
        self.location_hist.clear()
        self.dist_traveled = 0

        # ========= 1 加载 Town01 （如已在就跳过） =========
        current_map = self.world.get_map().name
        if current_map != "Town01":
            self.client.set_timeout(20.0)          # 临时放宽
            self.world = self.client.load_world("Town01")
            self.client.set_timeout(10.0)           # 立刻调回
            self.blueprint_library = self.world.get_blueprint_library()


        # ========= 2 固定起点 / 终点 =========
        spawn_points  = self.world.get_map().get_spawn_points()
        start_tf      = spawn_points[0]        # 起点（你可改索引）
        goal_tf       = spawn_points[15]       # 终点（你可改索引）

        # ========= 3 Spawn 车辆 =========
        prius_bp      = self.blueprint_library.filter('prius')[0]
        self.vehicle  = self.world.spawn_actor(prius_bp, start_tf)
        self.actor_list.append(self.vehicle)

        # ========= 4 绑定 RGB 摄像头 =========
        rgb_bp = self.blueprint_library.find('sensor.camera.rgb')
        rgb_bp.set_attribute('image_size_x', f'{self.im_width}')
        rgb_bp.set_attribute('image_size_y', f'{self.im_height}')
        rgb_bp.set_attribute('fov', '110')
        cam_tf = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.sensor = self.world.spawn_actor(rgb_bp, cam_tf, attach_to=self.vehicle)
        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data: self.process_img(data))

        # ========= 5 绑定碰撞检测 =========
        col_bp = self.blueprint_library.find('sensor.other.collision')
        self.col_sensor = self.world.spawn_actor(col_bp, cam_tf, attach_to=self.vehicle)
        self.actor_list.append(self.col_sensor)
        self.col_sensor.listen(lambda e: self.collision_hist.append(e))

        # ========= 6 让世界稳定一下 =========
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
        time.sleep(1.0)

        # ========= 7 规划路线 =========
        #from agents.navigation.global_route_planner import GlobalRoutePlanner
        #from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
        #from carla import GlobalRoutePlanner, GlobalRoutePlannerDAO
        wmap = self.world.get_map()                    # carla.Map
        grp  = GlobalRoutePlanner(wmap, 2.0)           # 采样分辨率 2 m
        self.route = [wp for wp, _ in grp.trace_route(start_tf.location, goal_tf.location)]

        # （可选）可视化路线
        #for wp in self.route[::5]:
            #self.world.debug.draw_string(
                #wp.transform.location + carla.Location(z=1),
                #'•', draw_shadow=False,
                #color=carla.Color(0, 0, 255), life_time=30.0)

        # ========= 8 等待第一帧图像 =========
        while self.front_camera is None:
            time.sleep(0.01)

        self.episode_start = time.time()
        return self.front_camera


    def collision_data(self, event):
        self.collision_hist.append(event)

    def process_img(self, image):
        i = np.array(image.raw_data)
        i2 = i.reshape((self.im_height, self.im_width, 4))
        i3 = i2[:, :, :3]
        if self.SHOW_CAM:
            cv2.imshow("", i3)
            cv2.waitKey(1)
        self.front_camera = i3

    def calc_distance(self, data):
        self.location_hist.append([data.latitude, data.longitude])
        if len(self.location_hist) > 1:
            x_dist = (self.location_hist[-1][0] - self.location_hist[0][0]) ** 2
            y_dist = (self.location_hist[-1][1] - self.location_hist[0][1]) ** 2
            self.dist_traveled = math.sqrt(x_dist + y_dist)


    def step(self, action):
        # ---------- ① 应用控制 ----------
        throttle_raw, steer, brake_raw = [float(x) for x in np.asarray(action).flatten()]
        #throttle = (throttle + 1) / 2.0
        # 只有明确“踩刹车”才启用（raw > 0.3）
        if brake_raw < 0.3:
            brake_raw = -1.0        # -1 → 缩放后 0
        else:
            brake_raw = 0.0
        throttle = 0.5 + 0.5 * (throttle_raw + 1)
        brake    = (brake_raw + 1) / 2.0
        #print(f"[TEST] action={action}, throttle={throttle:.3f}, steer={steer:.3f}, brake={brake:.3f}") #调试代码用
        self.vehicle.apply_control(carla.VehicleControl(
            throttle=throttle, steer=steer, brake=brake))

        if self.world.get_settings().synchronous_mode:
            self.world.tick()          # 推进一帧物理 & 传感器
        else:
            time.sleep(1.0 / FPS)      # 异步时让时间流逝

        # ---------- ② 基础信息 ----------
        v = self.vehicle.get_velocity()
        kph = 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)
        veh_tf   = self.vehicle.get_transform()

        # ---------- ③ 计算到路线最近点的横向偏差 ----------
        veh_loc = self.vehicle.get_location()
        nearest_wp = min(self.route, key=lambda wp: veh_loc.distance(wp.transform.location))
        deviation  = veh_loc.distance(nearest_wp.transform.location)   # 单位 m

        # ---------- ④ 奖励 ----------
        from settings import W_LANE, W_HEADING, W_SPEED, W_SMOOTH, W_PROGRESS
        

        #   4‑1 贴路线（deviation 越小越好）
        r_lane = 1.0 - min(deviation / 2.0, 1.0)              # ∈[0,1]

        #   4‑2 航向对齐
        wp_yaw      = nearest_wp.transform.rotation.yaw % 360
        veh_yaw     = veh_tf.rotation.yaw % 360
        heading_err = abs((veh_yaw - wp_yaw + 180) % 360 - 180) / 180  # ∈[0,1]
        r_heading   = 1.0 - heading_err                                # ∈[0,1]

        #   4‑3 速度（30 km/h 最佳）
        target = 30.0        # 目标巡航 30 km/h
        band   = 20.0        # 宽容带 ±20
        if kph<0.5:
            r_speed=-0.2
            #r_lane=0.0
            #r_heading=0.0
        elif kph < 5.0:
            #r_speed = -0.05                     # 强烈惩罚“趴窝”
            r_speed = (kph - 0.5) / 4.5 * 0.2 - 0.2
        else:
            r_speed = 1.0 - ((kph - target) / band) ** 2
        r_speed = np.clip(r_speed, -1.0, 1.0)

        #   4‑4 平滑控制（转向 / 油门变化小）
        if not hasattr(self, "prev_steer"):
            self.prev_steer, self.prev_throttle = steer, throttle
        steer_delta    = abs(steer    - self.prev_steer)
        throttle_delta = abs(throttle - self.prev_throttle)
        self.prev_steer, self.prev_throttle = steer, throttle
        r_smooth = 1.0 - (steer_delta + throttle_delta) / 2.0          # ∈[0,1]

        #   4‑5 前进进度（每 10 m 奖励 1 分）
        if not hasattr(self, "progress"):
            self.progress = 0.0
        self.progress += kph / 3.6 * (1.0 / FPS)                       # m
        r_progress = (self.progress // 10) * 2.0

        #   ——— 综合奖励 ———
        reward  = (
            W_LANE    * r_lane +
            W_HEADING * r_heading +
            W_SPEED   * r_speed +
            W_SMOOTH  * r_smooth +
            W_PROGRESS * r_progress
        )

        # ---------- ⑤ 终止 & 额外奖励/惩罚 ----------
        done = False
        if len(self.collision_hist):                 # 碰撞惩罚
            reward -= 10.0
            done = True
        elif veh_loc.distance(self.route[-1].transform.location) < 2.0:  # 到终点
            reward += 100.0
            done = True
        elif self.episode_start + SECONDS_PER_EPISODE < time.time():     # 超时
            reward -= 5.0
            done = True
        # ---------- ◎ 终止判定 ----------
        # 若 5 秒内速度持续 < 1 km/h，则视为“卡死”，强制结束
        if not hasattr(self, "_slow_timer"):
            self._slow_timer = 0.0          # ① 初始化 once

        if kph < 1.0:                       # ② 累计低速时长
            self._slow_timer += 1.0 / FPS
        else:
            self._slow_timer  = 0.0         # ③ 一旦速度恢复就清零计时

        if self._slow_timer > 10.0:          # ④ 连续 10 秒低速 → 终止
            reward -= 1.0
            done   = True
        #车载摄像头，测试时再使用，训练时注释掉
        #spectator   = self.world.get_spectator()          # 获取观察者
        #veh_tf      = self.vehicle.get_transform()        # 车辆当前位姿
        #cam_loc     = veh_tf.location + carla.Location(x=-8, z=6)   # 车后方 8 m、上方 6 m
        #spectator.set_transform(carla.Transform(cam_loc, veh_tf.rotation))
        
        return self.front_camera, reward, done, kph, deviation
    def _safe_destroy(self, actor):
        """
        安全地销毁 CARLA actor。若 actor 已失效则忽略 RuntimeError。
        """
        try:
            if actor is not None:
                actor.destroy()
        except RuntimeError:
            # actor 已被 CARLA 引擎销毁；忽略即可
            pass

