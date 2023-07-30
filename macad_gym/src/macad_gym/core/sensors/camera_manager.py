from __future__ import absolute_import
import os
import pygame
import weakref
import carla
import numpy as np
from enum import Enum
from macad_gym import RETRIES_ON_ERROR
from macad_gym.viz.logger import LOG


CAMERA_TYPES = Enum('CameraType', ['rgb',
                                   'depth_raw',
                                   'depth',
                                   'semseg_raw',
                                   'semseg'])


class CameraManager(object):
    """This class from carla, manual_control.py
    """

    def __init__(self, parent_actor, hud):
        self.image = None  # need image to encode obs.
        self.image_list = []  # for save images later.
        self.sensor = None
        self._surface = None
        self._parent = parent_actor
        self._hud = hud
        self._recording = False
        self._memory_record = False
        # supported through toggle_camera
        self._camera_transforms = [
            carla.Transform(carla.Location(x=1.8, z=1.7)),
            carla.Transform(carla.Location(x=-5.5, z=2.8),
                            carla.Rotation(pitch=-15)),
            carla.Transform(carla.Location(
                x=-2.0*(0.5 + self._parent.bounding_box.extent.x),
                y=0.0,
                z=2.0*(0.5 + self._parent.bounding_box.extent.z)),
                carla.Rotation(pitch=8.0))
        ]
        # 0 is dashcam view; 1 is tethered view; 2 for spring arm view (manual_control)
        self._transform_index = 0
        self._sensors = [
            ['sensor.camera.rgb', carla.ColorConverter.Raw, 'Camera RGB'],
            [
                'sensor.camera.depth', carla.ColorConverter.Raw,
                'Camera Depth (Raw)'
            ],
            [
                'sensor.camera.depth', carla.ColorConverter.Depth,
                'Camera Depth (Gray Scale)'
            ],
            [
                'sensor.camera.depth', carla.ColorConverter.LogarithmicDepth,
                'Camera Depth (Logarithmic Gray Scale)'
            ],
            [
                'sensor.camera.semantic_segmentation',
                carla.ColorConverter.Raw, 'Camera Semantic Segmentation (Raw)'
            ],
            [
                'sensor.camera.semantic_segmentation',
                carla.ColorConverter.CityScapesPalette,
                'Camera Semantic Segmentation (CityScapes Palette)'
            ], ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)']
        ]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self._sensors:
            bp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(hud.dim[0]))
                bp.set_attribute('image_size_y', str(hud.dim[1]))
            if item[0].startswith('sensor.camera.rgb'):
                bp.set_attribute('sensor_tick', str(0.0))
            item.append(bp)
        self._index = None
        self.callback_count = 0

    def destroy(self):
        if self.sensor is not None and self.sensor.is_alive:
            self.sensor.stop()
            self.sensor.destroy()
            self.image = None
            self.image_list.clear()
            self.sensor = None
            self._surface = None
            self.callback_count = 0

    def set_recording_option(self, option):
        """Set class vars to select recording method.

        Option 1: save image to disk while the program runs.(Default)
        Option 2: save to memory first. Save to disk when program finishes.

        Args:
            option (int): record method.

        Returns:
            N/A.
        """

        # TODO: The options should be more verbose. Strings instead of ints
        if option == 1:
            self._recording = True
        elif option == 2:
            self._memory_record = True

    def toggle_camera(self):
        self._transform_index = (self._transform_index + 1) % len(self._camera_transforms)
        self.set_sensor(self.index, self._transform_index, notify=False, force_respawn=True)

    def set_sensor(self, index, pos=0, notify=True, force_respawn=False):
        index = index % len(self._sensors)
        needs_respawn = True if self._index is None else (
            force_respawn or list(CAMERA_TYPES)[index] != list(CAMERA_TYPES)[self._index])
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.stop()
                self.sensor.destroy()
                self._surface = None
            self._transform_index = pos % len(self._camera_transforms)
            for i in range(RETRIES_ON_ERROR):
                self.sensor = self._parent.get_world().try_spawn_actor(
                    self._sensors[index][-1],
                    self._camera_transforms[self._transform_index],
                    attach_to=self._parent,
                    attachment_type=carla.AttachmentType.Rigid if pos != 2 else carla.AttachmentType.SpringArm)
                if self.sensor is None:
                    LOG.camera_manager_logger.error(f"Spawn RGBCamera faild, "
                                f"parent actor:{self._parent.type_id} {self._parent.id} "
                                f"retry times:{i}")
                else:
                    break
            # We need to pass the lambda a weak reference to self to avoid
            # circular reference.
            weak_self = weakref.ref(self)
            # Some sensors may require to send data asynchronously，
            # camera sensors send the images from the render thread.
            self.sensor.listen(
                lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            self._hud.notification(self._sensors[index][2])
        self._index = index

    def next_sensor(self):
        self.set_sensor(self._index + 1)

    def toggle_recording(self):
        self._recording = not self._recording
        self._hud.notification('Recording %s' %
                               ('On' if self._recording else 'Off'))

    def render(self, display, render_pose=(0, 0)):
        if self._surface is not None:
            display.blit(self._surface, render_pose)

    @staticmethod
    def _parse_image(weak_self, image):
        if not weak_self():
            return
        self = weak_self()
        
        self.image = image
        self.callback_count += 1

        if self._sensors[self._index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 3), 3))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self._hud.dim) / 100.0
            lidar_data += (0.5 * self._hud.dim[0], 0.5 * self._hud.dim[1])
            lidar_data = np.fabs(lidar_data)
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self._hud.dim[0], self._hud.dim[1], 3)
            lidar_img = np.zeros(lidar_img_size)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self._surface = pygame.surfarray.make_surface(lidar_img)
        else:
            image.convert(self._sensors[self._index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self._surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if self._recording:
            image_dir = os.path.join(
                LOG.log_dir, 'images/{}/%04d.png'.format(self._parent.id) %
                image.frame_number)
            image.save_to_disk(image_dir)  # , env.cc
            # image.save_to_disk('_out/%08d' % image.frame_number)
        elif self._memory_record:
            self.image_list.append(image)
        else:
            pass
