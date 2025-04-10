#!/usr/bin/env python3 
import numpy as np
import cv2
import torch
import time
from ultralytics import YOLO
import pykinect_azure as pykinect
from pykinect_azure import K4A_CALIBRATION_TYPE_COLOR, K4A_CALIBRATION_TYPE_DEPTH, k4a_float2_t, K4A_FRAMES_PER_SECOND_30, K4A_WIRED_SYNC_MODE_STANDALONE


class YoloResult:
    def __init__(self, name, box, x, y, conf, depth_image, depth, world_coords) -> None:
        self.name = name
        self.box = box
        self.x = x
        self.y = y
        self.conf = conf
        self.depth_image = depth_image
        self.depth = depth
        self.world_coords = world_coords  # [x, y, z] in camera coordinates
        
    def __str__(self):
        return f'name:{self.name}, box:{self.box}, x:{self.x}, y:{self.y}, conf:{self.conf:.2f}' + \
               (f', depth:{self.depth:.2f}m, world:{self.world_coords}' if self.depth is not None else '')


class Detector:
    def __init__(self) -> None:
        self.model = YOLO('./model/yolo11m.pt')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        self.model.to(self.device)

        # Kinect camera intrinsic parameters (1080P)
        self.K_kinect = np.array([926.658159, 0.000000, 969.015082,
                                 0.000000, 927.187798, 549.125794,
                                 0.000000, 0.000000, 1.000000]).reshape(3, 3)
        
        # For storing all detection results
        self.all_detections = []

    def detect(self, camera='cam', pattern='realtime', target='bottle', depth=True, range=0.5):
        """
        Detection main function
        
        Args:
            camera: 'k4a' or 'kinect' for Kinect camera,
                   'realsense' for Intel RealSense,
                   'astra' for Orbbec Astra S,
                   'cam' for built-in webcam
            pattern: 'realtime' for continuous detection (press q to stop),
                     'find' to search for target until found with high confidence
            target: target object class to look for
            depth: whether to use depth information
            range: center range ratio for target detection (0-1)
        
        Returns:
            List of dictionaries containing detection results with coordinates
        """
        results_detect = []
        
        if camera == 'k4a' or camera == 'kinect':
            pykinect.initialize_libraries()
            device_config = pykinect.default_configuration
            device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_MJPG
            device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_1080P
            device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED
            device_config.camera_fps = K4A_FRAMES_PER_SECOND_30
            device_config.wired_sync_mode = K4A_WIRED_SYNC_MODE_STANDALONE
            device_config.synchronized_images_only = True

            device = pykinect.start_device(config=device_config)
            calibration = device.get_calibration(
                depth_mode=device_config.depth_mode,
                color_resolution=device_config.color_resolution
            )
            transformation = pykinect.Transformation(calibration)

            while True:
                capture = device.update()
                ret, color_frame = capture.get_color_image()
                
                if depth:
                    retd, depth_image = capture.get_transformed_depth_image()
                    if not ret or not retd:
                        continue
                    self.depth_image = depth_image  # 保存深度图像供pred方法使用
                else:
                    if not ret:
                        continue

                self.color_frame = color_frame
                yoloresults = self.pred(depth, target)
                
                if not yoloresults:
                    print("No objects detected, continuing...")
                    continue

                height, width = color_frame.shape[:2]
                resolution = [width, height]
                self.show(resolution, range)

                # Process detections and calculate coordinates
                frame_results = []
                for yolor in yoloresults:
                    result_data = {
                        'class': yolor.name,
                        'confidence': float(yolor.conf),
                        'image_coords': (int(yolor.x), int(yolor.y)),
                        'bounding_box': yolor.box
                    }
                    
                    if depth:
                        # Calculate depth and world coordinates
                        depth_val = depth_image[int(yolor.y), int(yolor.x)] * 0.001  # Convert to meters
                        world_coords = self.calculate_world_coordinates(
                            yolor.x, yolor.y, depth_val, camera
                        )
                        
                        result_data.update({
                            'depth': depth_val,
                            'world_coords': world_coords.tolist() if world_coords is not None else None
                        })
                        
                        yolor.depth = depth_val
                        yolor.world_coords = world_coords
                    
                    frame_results.append(result_data)

                print(frame_results)
                
                self.all_detections.append(frame_results)

                # Check if we should stop based on pattern
                if self.judge(pattern, yoloresults, target, width, range):
                    if pattern == 'find':
                        # For 'find' mode, we return the target detections
                        target_detections = [
                            res for res in frame_results 
                            if res['class'] == target and self.judge_range(res['image_coords'][0], width, range)
                        ]
                        results_detect.extend(target_detections)
                        cv2.imwrite("target.jpg", self.color_frame)
                        if target_detections:  # If we found our target, we can break
                            break
                
                key_pressed = cv2.waitKey(10)
                if key_pressed in [ord('q'), 27]:  # q or ESC to quit
                    break
            
            device.stop_cameras()
            device.close()

        cv2.destroyAllWindows()
        
        # Return all detections from the last frame if not in 'find' mode
        if pattern != 'find' and self.all_detections:
            return self.all_detections[-1]
        return results_detect

    # def pred(self, depth_, target_):
    #     results = self.model(self.color_frame)
    #     self.color_frame = results[0].plot()
    #     model_names = results[0].names
    #     yoloresults = []

    #     boxes = results[0].boxes

    #     for i in range(len(boxes)):
    #         box = boxes[i]
    #         x1, y1, x2, y2 = map(int, box.xyxy[0])
    #         conf = float(box.conf[0])
    #         cls = int(box.cls[0])
    #         class_name = model_names[cls]

    #         center_x = (x1 + x2) / 2
    #         center_y = (y1 + y2) / 2
    #         depth =
    #         world_coords = 

    #         yoloresult = YoloResult(
    #             name=class_name,
    #             box=[x1, y1, x2, y2],
    #             x=center_x,
    #             y=center_y,
    #             conf=conf,
    #             depth_image=None,
    #             depth=None,
    #             world_coords=None
    #         )
            
    #         if depth_:
    #             # if target_ and yoloresult.name == target_:
    #             yoloresults.append(yoloresult)
    #         # else:
    #         #     yoloresults.append(yoloresult)

    #     return yoloresults
    def pred(self, depth_, target_):
        """
        执行目标检测并返回带有深度信息的结果
        
        Args:
            depth_: 是否使用深度信息
            target_: 目标类别名称(用于过滤)
            
        Returns:
            List[YoloResult]: 检测结果列表，包含深度和世界坐标(如果启用)
        """
        # 执行YOLO检测
        results = self.model(self.color_frame)
        self.color_frame = results[0].plot()  # 绘制检测结果到图像
        model_names = results[0].names
        yoloresults = []

        # 获取当前帧的深度图像(如果启用深度)
        if depth_:
            # 注意: 需要在主循环中获取并保存depth_image
            depth_image = getattr(self, 'depth_image', None)
            if depth_image is None:
                print("Warning: Depth image not available")
                depth_ = False  # 强制禁用深度处理

        # 处理每个检测结果
        for result in results[0]:
            box = result.boxes
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            class_name = model_names[cls]

            # 计算边界框中心点
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # 初始化深度和世界坐标
            depth_val = None
            world_coords = None

            # 如果启用深度处理
            if depth_:
                try:
                    # 获取中心点的深度值(毫米转米)
                    depth_val = depth_image[int(center_y), int(center_x)] * 0.001
                    
                    # 计算世界坐标
                    if depth_val > 0:  # 有效的深度值
                        point_image = np.array([center_x, center_y, 1])
                        world_coords = depth_val * np.linalg.inv(self.K_kinect).dot(point_image)
                except Exception as e:
                    print(f"Depth processing error: {e}")
                    depth_val = None
                    world_coords = None

            # 创建结果对象
            yoloresult = YoloResult(
                name=class_name,
                box=[x1, y1, x2, y2],
                x=center_x,
                y=center_y,
                conf=conf,
                depth_image= depth_image,
                depth=depth_val,
                world_coords=world_coords
            )
            
            # 如果指定了目标类别，只返回匹配的结果
            if target_:
                if yoloresult.name == target_:
                    yoloresults.append(yoloresult)
            else:
                yoloresults.append(yoloresult)

        return yoloresults

    def show(self, resolution, range):
        """
        Display the detection results with guide lines
        """
        width, height = resolution
        cv2.line(self.color_frame, 
                (int(width * 0.5 * (1 - range)), 0), 
                (int(width * 0.5 * (1 - range)), height),
                (0, 255, 0), 2, 4)
        cv2.line(self.color_frame, 
                (int(width * 0.5 * (1 + range)), 0), 
                (int(width * 0.5 * (1 + range)), height),
                (0, 255, 0), 2, 4)
        
        cv2.imshow('yolo', self.color_frame)

    def judge(self, pattern, yolors, target=None, resolution=640, range=0.8):
        """
        Determine whether to continue detection based on pattern
        
        Args:
            pattern: detection mode ('realtime', 'find', etc.)
            yolors: list of YoloResult objects
            target: target class name
            resolution: image width
            range: center range ratio
        
        Returns:
            bool: whether to continue detection
        """
        if pattern == "realtime":
            return True  # Continue until user stops
        elif pattern == 'find':
            for yolor in yolors:
                if target == yolor.name and self.judge_range(yolor.x, resolution, range):
                    return True  # Found target in center range
            return False  # Continue searching
        return True  # Default to continue

    def judge_range(self, x, resolution=640, range=0.8):
        """
        Check if point is within the center range
        
        Args:
            x: x-coordinate to check
            resolution: image width
            range: center range ratio (0-1)
        
        Returns:
            bool: whether point is in range
        """
        left = resolution * 0.5 * (1 - range)
        right = resolution * 0.5 * (1 + range)
        return left <= x <= right

    def calculate_world_coordinates(self, x, y, depth, camera):
        """
        Calculate world coordinates from image coordinates and depth
        
        Args:
            x, y: image coordinates
            depth: depth value in meters
            camera: camera type (for correct intrinsics)
        
        Returns:
            numpy array: [x, y, z] in camera coordinates
        """
        if camera == 'k4a' or camera == 'kinect':
            K = self.K_kinect
        else:
            return None
        
        if depth <= 0:  # Invalid depth
            return None
            
        # Convert to camera coordinates
        point_image = np.array([x, y, 1])
        point_camera = depth * np.linalg.inv(K).dot(point_image)
        return point_camera


if __name__ == "__main__":
    detector = Detector()
    results = detector.detect(
        camera='k4a',
        pattern='realtime',  # Change to 'find' if looking for specific target
        target='bottle',     # Only used when pattern='find'
        depth=True,
        range=0.8
    )
    