# AgileX Bunker 边缘部署

Bunker 使用 AgileX 的 [ugv_sdk](https://github.com/agilexrobotics/ugv_sdk) 和 [bunker_ros2](https://github.com/agilexrobotics/bunker_ros2)。部署脚本复用 Go2 的 RealSense 发布器和 InternNav HTTP 客户端。当前服务端返回轨迹或离散动作，边缘客户端的 MPC/PID 根据轨迹和里程计计算速度。官方驱动把速度交给底盘控制器；底盘控制器负责驱动履带。

| 数据 | ROS 2 话题 | 连接 |
|---|---|---|
| 彩色图像 | `/camera/camera/color/image_raw` | RealSense → HTTP 客户端 |
| 对齐深度 | `/camera/camera/aligned_depth_to_color/image_raw` | RealSense → HTTP 客户端 |
| 里程计 | `/odom_bridge` | `bunker_base_node` → HTTP 客户端 |
| 导航速度 | `/cmd_vel_bridge` | HTTP 客户端 → Bunker 桥接 |
| 底盘速度 | `/internnav/bunker/cmd_vel` | Bunker 桥接 → 官方驱动（重映射其 `/cmd_vel`） |

桥接只转发 `Twist.linear.x`（m/s）和 `Twist.angular.z`（rad/s）。官方驱动根据底盘反馈的速度积分发布里程计，履带打滑会造成漂移。已有融合定位时，可用 `base_odom_topic:=/bunker/wheel_odom` 将底盘里程计发布到单独的话题，再让定位节点发布 `/odom_bridge`。若定位节点也发布 `odom → base_link`，还需避免重复发布 TF，例如把驱动的 `odom_frame` 和 `base_frame` 设置为单独的轮速参考系。

## 1. 安装官方底盘驱动

以下以 Ubuntu 24.04 / ROS 2 Jazzy 为例。厂商 ROS 2 仓库的分支叫 `humble`，Jazzy 需从源码构建，不能直接加载用 Humble 编译的驱动。

```bash
source /opt/ros/jazzy/setup.bash
sudo apt install build-essential cmake git libasio-dev python3-colcon-common-extensions \
  ros-jazzy-ros-base ros-jazzy-tf2-geometry-msgs ros-jazzy-cv-bridge ros-jazzy-message-filters

mkdir -p ~/bunker_ws/src
cd ~/bunker_ws
git clone --depth 1 --branch main https://github.com/agilexrobotics/ugv_sdk.git ugv_sdk
git clone --depth 1 --branch humble https://github.com/agilexrobotics/bunker_ros2.git src/bunker_ros2
cmake -S ugv_sdk -B sdk_build -DCMAKE_INSTALL_PREFIX="$PWD/sdk_install"
cmake --build sdk_build -j2
cmake --install sdk_build
colcon build --base-paths src --packages-up-to bunker_base \
  --cmake-args -Dugv_sdk_DIR="$PWD/sdk_install/lib/cmake/ugv_sdk" -DBUILD_TESTING=OFF
```

本机通过编译的版本为 `ugv_sdk` 的 `f2704eacdc90357078cd93ec60aae08bb4baab35` 和 `bunker_ros2` 的 `c4737f249129e88c8e9e0bfeb3af81b498a0ebbe`。若分支更新后编译失败，可在对应仓库检出这两个提交再构建（浅克隆未包含旧提交时先 `git fetch --unshallow`）。

启动相机的 Python 还需能导入 `pyrealsense2` 和 `cv_bridge`。若用 Python 3.12 虚拟环境运行相机和客户端，可在已激活的环境中安装 `uv pip install 'numpy<2' pyyaml pyrealsense2 requests pillow scipy casadi`；NumPy 1.x 用于兼容 Jazzy 的 Ubuntu 24.04 二进制 `cv_bridge`。如果板载设备没有对应的 `pyrealsense2` wheel，按 [RealSense SDK 文档](https://github.com/IntelRealSense/librealsense) 构建对应的 Python 绑定。

云端继续使用现有的 `scripts/realworld/http_internvla_server.py`。部署前按实测标定检查其中的 `args.camera_intrinsic` 和 `camera_pose`，并把示例 `instruction` 换成实际任务。默认图像为 640×480；换相机或分辨率时，内参也需对应更新。

## 2. 检查启动参数

从 `InternNav-RL` 根目录运行。这条命令只列出参数，不启动节点，也不需要 CAN 或相机：

```bash
bash InternNav-deploy/bunker/start_robot.sh --show-args
```

## 3. 连接实车后启动

确认铭牌型号和固件，并按 [AgileX 手册](https://github.com/agilexrobotics/AgileX-Robotics-all-products-user-manuals)配置 CAN。厂商示例使用 `can0`、500 kbit/s；若你的型号或网卡不同，以手册和现场配置为准。初次测试保留遥控接管和急停。不要同时启动 Go2 的脚本或其他同名话题发布者。

终端 1，从 `InternNav-RL` 根目录启动底盘、桥接和相机：

```bash
source /opt/ros/jazzy/setup.bash
source ~/bunker_ws/install/setup.bash
bash InternNav-deploy/bunker/start_robot.sh can_interface:=can0
```

Bunker Mini 可加 `is_bunker_mini:=true`，该版本驱动用它标记启动日志，CAN 协议仍由驱动自动检测。若 RealSense 依赖在虚拟环境内，启动命令追加 `camera_python:=/绝对路径/.venv/bin/python`。相机已经由其他节点发布上述图像话题时，用 `start_camera:=false`。

终端 2，先确认 `/odom_bridge` 和 RGB/深度话题在更新，再启动原 HTTP 客户端：

```bash
source /opt/ros/jazzy/setup.bash
python3 scripts/realworld/http_internvla_client.py --ros-args \
  -p server_url:=http://SERVER_IP:5801/eval_dual -p max_data_age:=2.0
```

`max_data_age` 以秒计。RGB-D、里程计或推理结果过期时，客户端输出零速；默认启动参数还会限制速度在 0.3 m/s、0.4 rad/s，并在 0.5 秒未收到速度时发布零速。按实车和实测推理延迟调整这些参数。软件超时依赖进程和通信正常工作，不能代替底盘急停。
