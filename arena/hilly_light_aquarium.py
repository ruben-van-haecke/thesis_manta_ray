import os
from pathlib import Path
import  matplotlib.pyplot as plt

import numpy as np
from PIL import Image
from dm_control import mjcf
from dm_control.composer import Arena
from dm_control.mujoco.wrapper import mjbindings
from scipy.ndimage import gaussian_filter
from transforms3d.euler import euler2quat

from thesis_manta_ray.arena.obstacle import Obstacle
from thesis_manta_ray.arena.utils import colors
from thesis_manta_ray.arena.utils.colors import rgba_sand
from thesis_manta_ray.arena.utils.noise import generate_perlin_noise_map
from thesis_manta_ray.arena.entities.target import Target
from thesis_manta_ray.arena.entities.parkour_obstacles import CilinderParkour

from fiblat import sphere_lattice


class OceanArena(Arena):
    assets_directory = str(Path(__file__).parent / "assets")   
    BASE_ELEVATION = 0.5
    MAX_ELEVATION = 1.5
    BASE_LIGHT = 0.2
    MAX_LIGHT = 1.0
    MAX_NOISE_WEIGHT = 1.0
    needs_collision = []

    def _build(
            self,
            initial_morphology_position: tuple[int, int] = (0, 0),
            name='hilly_light_aquarium',
            env_id: int = 0,
            size=(10, 10, 10),
            light_texture: bool = False,
            light_noise: bool = False,
            targeted_light: bool = False,
            hilly_terrain: bool = False,
            random_current: bool = False,
            random_friction: bool = False,
            random_obstacles: bool = False,
            task_mode: str = "no_target",
            ) -> None:
        """
        args:
            task_mode: str, "no_target": no obstacles
                            "random_target": one random target
                            "grid": a raster in front of the robot in which a target
                            "parkour": parkour to follow
        """
        assert task_mode in ["no_target", "random_target", "grid", "parkour"], "task_mode is not valid"

        super()._build(name=name)
        self._initial_morphology_position = np.array(initial_morphology_position)
        self._dynamic_assets_identifier = env_id
        self.size = np.array(size)
        self._light_texture = light_texture
        self._light_noise = light_noise
        self._targeted_light = targeted_light
        self._hilly_terrain = hilly_terrain
        self._random_current = random_current
        self._random_friction = random_friction
        self._random_obstacles = random_obstacles
        self._task_mode = task_mode

        self._configure_assets_directory()
        self._generate_random_height_and_light_maps()
        self._configure_cameras()
        self._configure_lights()
        self._configure_sky()
        self._build_ground()
        self._build_walls()
        if self._task_mode == "random_target":    # one random obstacle
            self.target = self._attach_target()
        elif self._task_mode == "grid":  # raster of obstacles of which one is choosen
            self.target = self._attach_target()
            points = 101
            radius = 5
            self._grid_coordinates = np.empty((3, points))
            sphere = sphere_lattice(3, points)
            self._grid_coordinates = radius * sphere  # x
            self._grid_coordinates = self._grid_coordinates[self._grid_coordinates[:, 0] < 0]
            # # Create a figure and a 3D subplot
            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')

            # # Plot the surface
            # ax.scatter(self._grid_coordinates[:, 0], 
            #                 self._grid_coordinates[:, 1],
            #                 self._grid_coordinates[:, 2],
            #                 marker='o')

            # # Set labels
            # ax.set_xlabel('X')
            # ax.set_ylabel('Y')
            # ax.set_zlabel('Z')

            # # Show the plot
            # plt.show()

        elif self._task_mode == "parkour":  # parkour
            self._obstacles_traject = self._build_obstacle_course()
        # self._build_obstacles()
        # self._build_current_arrow()
        self._configure_water()

        self._previous_light_shift_time = 0
    
    def _build_obstacle_course(
            self,
            ) -> None:
        # build obstacles
        obstacle_course = []
        obstacle_course.append(CilinderParkour())
        for entity in obstacle_course:
            self.attach(entity)
        return obstacle_course
    
    def _build_path(
            self,
            ) -> None:
        return 
    
    def _attach_target(
            self
            ) -> None: 
        target = Target()
        self.attach(target)
        return target
    
    def randomize_target_location(
            self,
            physics: mjcf.Physics,
            distance_from_origin: float
            ) -> None:
        angle1, angle2 = 2*np.pi*np.random.random(), 2*np.pi*np.random.random()
        position = distance_from_origin * np.array([np.cos(angle1), np.sin(angle1), np.abs(np.cos(angle2))])
        self.target.set_pose(
            physics=physics, position=position
            )
    
    def target_grid_location(
            self,
            physics: mjcf.Physics,
            index: int,
            ) -> None:
        self.target.set_pose(
                physics=physics, position=self._grid_coordinates[index]
                )

    def _configure_assets_directory(
            self
            ) -> None:
        self.mjcf_model.compiler.texturedir = self.assets_directory
        self.mjcf_model.compiler.meshdir = self.assets_directory
        self.mjcf_model.compiler.assetdir = self.assets_directory

    @property
    def _light_map_asset_path(
            self
            ) -> str:
        return f"{str(Path(__file__).parent)}/light_map/lightmap_{self._dynamic_assets_identifier}.png"

    @property
    def _heightmap_asset_path(
            self
            ) -> str:
        return f"{str(Path(__file__).parent)}/height_map/heightmap_{self._dynamic_assets_identifier}.png"

    def _generate_random_height_and_light_maps(
            self,
            save_assets: bool = True
            ) -> None:
        width, height = 256, 128
        starting_zone_width = width // 5

        # Generate hills
        if self._hilly_terrain:
            heightmap = generate_perlin_noise_map(width=width, height=height, octaves=10)

            # weight hills to start out flat
            heightmap[:, :starting_zone_width] = 0
            hill_weights = np.zeros((height, width))
            for i, weight in enumerate(np.linspace(0, self.MAX_NOISE_WEIGHT, width - starting_zone_width)):
                hill_weights[:, i + starting_zone_width] = weight

            self._heightmap = heightmap * hill_weights
        else:
            self._heightmap = np.zeros((height, width))

        if self._targeted_light:
            lightmap = np.zeros_like(self._heightmap)
        else:
            # Linear transition from bright to dark
            lightmap = np.zeros_like(self._heightmap)
            for x, brightness in enumerate(np.linspace(1, 0, width)):
                lightmap[:, x] = brightness

        if self._light_noise:
            if self._hilly_terrain:
                # Use the same noise as the hilly terrain
                light_noise = np.copy(self._heightmap)
                # Constant weights as hills already increase over width
                light_noise_weights = np.ones((height, width))
            else:
                light_noise = generate_perlin_noise_map(width=width, height=height, octaves=10)
                light_noise = 1 - 2 * light_noise  # Rescale to [-1, 1] range

                # increasing weights over width
                light_noise_weights = np.zeros_like(light_noise)
                for x, weight in enumerate(np.linspace(0, self.MAX_NOISE_WEIGHT, width - starting_zone_width)):
                    light_noise_weights[:, x + starting_zone_width] = weight

            light_noise_weights[:, :starting_zone_width] = 0.0

            # Light noise interval from [0, 1] to [1 - noise, 1 + noise]
            light_noise = 1 + light_noise_weights * light_noise
            lightmap *= light_noise

            lightmap = gaussian_filter(lightmap, mode="wrap", sigma=3)

            # normalize light map
            lightmap = (lightmap - np.min(lightmap)) / (np.max(lightmap) - np.min(lightmap))

        self.lightmap = self.BASE_LIGHT + (self.MAX_LIGHT - self.BASE_LIGHT) * lightmap

        # color
        lightmap = np.stack((self.lightmap,) * 3, axis=-1)
        light_color_map = lightmap * rgba_sand[:3]
        self._color_lightmap = (light_color_map * 255).astype(np.uint8)

        if save_assets:
            hm_im = Image.fromarray(self._heightmap * 255).convert('L')
            hm_im.save(self._heightmap_asset_path)
            lm_im = Image.fromarray(self._color_lightmap).convert('RGB')
            lm_im.save(self._light_map_asset_path)
    @property
    def mjcf_root(self):
        return self._mjcf_root

    def _configure_cameras(
            self
            ):
        # position = self.
        self._mjcf_root.worldbody.add(
                'camera', name='top_camera', pos=[-1, -1, 20], quat=[1, 0, 0, 0], )

    def _configure_lights(
            self
            ):
        self.mjcf_model.worldbody.add(
                'light', pos=[0, 0, 20], directional=True, dir=[0, 0, -0.5], diffuse=[0.1, 0.1, 0.1], castshadow=True
                )

    def _configure_sky(
            self
            ) -> None:
        # white sky
        self._mjcf_root.asset.add(
                'texture', type='skybox', builtin='flat', rgb1='1.0 1.0 1.0', rgb2='1.0 1.0 1.0', width=200, height=200
                )

    def _build_ground(
            self
            ) -> None:
        if self._light_texture:
            self._ground_texture = self._mjcf_root.asset.add(
                    'texture', type='2d', name='light_gradient_ground_plane', file=self._light_map_asset_path
                    )
            ground_material = self._mjcf_root.asset.add(
                    'material', name='groundplane', reflectance=0.0, texture=self._ground_texture
                    )
        else:
            ground_texture = self._mjcf_root.asset.add(
                    'texture',
                    rgb1=[.2, .3, .4],
                    rgb2=[.1, .2, .3],
                    type='2d',
                    builtin='checker',
                    name='groundplane',
                    width=200,
                    height=200,
                    mark='edge',
                    markrgb=[0.8, 0.8, 0.8]
                    )
            ground_material = self._mjcf_root.asset.add(
                    'material', name='groundplane', texrepeat=[2, 2],  # Makes white squares exactly 1x1 length units.
                    texuniform=True, reflectance=0.0, texture=ground_texture
                    )

        self._ground_hfield = self._mjcf_root.asset.add(
                "hfield",
                name="heightmap",
                file=self._heightmap_asset_path,
                size=tuple(self.size[:2]) + (self.MAX_ELEVATION, self.BASE_ELEVATION)
                )

        # Build groundplane.
        self._ground_geom = self._mjcf_root.worldbody.add(
                'geom', type='hfield', name='height_map_geom', hfield="heightmap", material=ground_material
                )
        self.needs_collision.append(self._ground_geom)

    def _build_walls(
            self
            ) -> None:
        # build walls
        wall_thickness = 0.1
        wall_height = (self.BASE_ELEVATION + self.MAX_ELEVATION + wall_thickness)
        wall_z_pos = wall_height - self.BASE_ELEVATION
        wall_rgba = np.asarray([115, 147, 179, 50]) / 255
        self.needs_collision.append(
                self._mjcf_root.worldbody.add(
                        'geom',
                        type='box',
                        name='north_wall',
                        size=[self.size[0], wall_thickness, wall_height],
                        pos=[0.0, self.size[1] + wall_thickness, wall_z_pos],
                        rgba=wall_rgba
                        )
                )
        self.needs_collision.append(
                self._mjcf_root.worldbody.add(
                        'geom',
                        type='box',
                        name='south_wall',
                        size=[self.size[0], wall_thickness, wall_height],
                        pos=[0.0, -self.size[1] - wall_thickness, wall_z_pos],
                        rgba=wall_rgba
                        )
                )
        self.needs_collision.append(
                self._mjcf_root.worldbody.add(
                        'geom',
                        type='box',
                        name='east_wall',
                        size=[wall_thickness, self.size[1] + 2 * wall_thickness, wall_height],
                        pos=[-self.size[0] - wall_thickness, 0.0, wall_z_pos],
                        rgba=wall_rgba
                        )
                )
        self.needs_collision.append(
                self._mjcf_root.worldbody.add(
                        'geom',
                        type='box',
                        name='west_wall',
                        size=[wall_thickness, self.size[1] + 2 * wall_thickness, wall_height],
                        pos=[self.size[0] + wall_thickness, 0.0, wall_z_pos],
                        rgba=wall_rgba
                        )
                )

    def _get_obstacle_positions(
            self
            ) -> np.ndarray:
        offset_between_obstacles = 2

        positions = []
        arena_width, arena_height = self.size
        for x in range(-arena_width + 1, arena_width, offset_between_obstacles):
            for y in range(-arena_height + 1, arena_height, offset_between_obstacles):
                positions.append([x, y, 0])

        positions = np.stack(positions).astype(float)
        return positions

    def _build_obstacles(
            self
            ) -> None:
        if self._random_obstacles:
            num_obstacles = int(self._get_obstacle_positions().size // 3)
            self._obstacles = [Obstacle(size=[0.2, 0.75]) for _ in range(num_obstacles)]
            for obstacle in self._obstacles:
                self.attach(obstacle)

    def _randomize_obstacles(
            self,
            physics: mjcf.Physics
            ) -> None:
        if self._random_obstacles:
            positions = self._get_obstacle_positions()
            positions[:, 1] += brb.brb_random_state.uniform(-0.8, 0.8, len(positions))

            # Remove positions that are too close to morphology position
            mask = np.linalg.norm(
                    positions[:, :2] - self._initial_morphology_position,
                    axis=1
                    ) > 2

            # Remove positions that are too close to arena bounds
            min_x, max_x = -self.size[0] + 1, self.size[0] - 1
            min_y, max_y = -self.size[1] + 1, self.size[1] - 1
            mask &= ((min_x <= positions[:, 0]) & (positions[:, 0] <= max_x) &
                     (min_y <= positions[:, 1]) & (positions[:, 1] <= max_y))
            # Removal means placing them out of bounds
            positions[~mask] = [100, 100, 100]

            for pos, obstacle in zip(positions, self._obstacles):
                y_shift = brb.brb_random_state.uniform(-0.8, 0.8)
                pos[1] += y_shift
                obstacle.set_pose(physics=physics,
                                  position=pos)

    def _build_current_arrow(
            self
            ) -> None:
        if self._random_current:
            self.mjcf_model.asset.add(
                    "mesh", name=f"direction_arrow", file=f"{self.assets_directory}/arrow.stl", scale=0.005 * np.ones(3)
                    )

            self._current_arrow = self._mjcf_root.worldbody.add(
                    'geom',
                    name=f"drag_direction_arrow",
                    type="mesh",
                    mesh=f"direction_arrow",
                    pos=[0.0, self.size[1], 6.0],
                    rgba=colors.rgba_red, )

    def _configure_water(
            self
            ):
        self.mjcf_model.option.density = 1000   # 1000 is default
        self.mjcf_model.option.viscosity = 0.0009   # 0. is default
        self.mjcf_model.option.gravity = np.array([0, 0, 0])    # [0, 0, -9.81] is default
        self.mjcf_model.option.integrator = "implicit"   # "Euler" is default
        # self.mjcf_model.option.flag.energy = "enable"    # "disable" is default, used for monitoring energy for reward function? -> no, only kinetic energy
        # self.mjcf_model.option.timestep = 0.01    # 0.02 is default

    def _update_ground_hfield(
            self,
            physics: mjcf.Physics
            ) -> None:
        if self._hilly_terrain:
            hfield = physics.bind(self._ground_hfield)
            h, w, adr = hfield.nrow, hfield.ncol, hfield.adr
            size = h * w

            physics.model.hfield_data[adr: adr + size] = self._heightmap.flatten()

            if physics.contexts:
                with physics.contexts.gl.make_current() as ctx:
                    ctx.call(
                            mjbindings.mjlib.mjr_uploadHField,
                            physics.model.ptr,
                            physics.contexts.mujoco.ptr,
                            hfield.element_id
                            )

    def _update_ground_texture(
            self,
            physics: mjcf.Physics
            ) -> None:
        if (self._light_noise and self._light_texture) or self._targeted_light:
            texture = physics.bind(self._ground_texture)
            h, w, adr = texture.height, texture.width, texture.adr
            size = h * w * 3

            physics.model.tex_rgb[adr: adr + size] = self._color_lightmap.flatten()

            if physics.contexts:
                with physics.contexts.gl.make_current() as ctx:
                    ctx.call(
                            mjbindings.mjlib.mjr_uploadTexture,
                            physics.model.ptr,
                            physics.contexts.mujoco.ptr,
                            texture.element_id
                            )

    def shift_lightmap(
            self,
            physics: mjcf.Physics
            ) -> bool:
        if physics.time() - self._previous_light_shift_time > 0.5:
            self.lightmap = np.roll(self.lightmap, shift=1, axis=0)
            self._color_lightmap = np.roll(self._color_lightmap, shift=1, axis=0)
            self._update_ground_texture(physics=physics)
            self._previous_light_shift_time = physics.time()
            return True
        return False

    def targeted_lightmap(
            self,
            physics: mjcf.Physics,
            target_xy_world: np.ndarray
            ) -> None:
        # Transform world coordinates to normalized coordinates in [0, 1] based on arena size
        target_xy_normalized = (target_xy_world + self.size) / (2 * self.size)
        # Flip y axis
        target_xy_normalized[1] = 1 - target_xy_normalized[1]

        # Transform normalized coordinates to lightmap pixel coordinates
        lightmap_size = self.lightmap.shape
        target_yx_normalized = target_xy_normalized[::-1]
        target_yx_lightmap = lightmap_size * target_yx_normalized

        # Get a circular mask
        center = target_yx_lightmap
        radius = 4
        y, x = np.ogrid[:self.lightmap.shape[0], :self.lightmap.shape[1]]
        mask = (y - center[0]) ** 2 + (x - center[1]) ** 2 <= radius ** 2

        self.lightmap[:] = self.BASE_LIGHT
        self.lightmap[mask] = 1.0

        self.lightmap = gaussian_filter(self.lightmap, mode="reflect", sigma=3)

        lightmap = np.stack((self.lightmap,) * 3, axis=-1)
        light_color_map = lightmap * rgba_sand[:3]
        self._color_lightmap = (light_color_map * 255).astype(np.uint8)
        self._update_ground_texture(physics=physics)

    def _randomize_ground_friction(
            self,
            physics: mjcf.Physics
            ) -> None:
        if self._random_friction:
            random_sliding_friction = brb.brb_random_state.uniform(low=0.5, high=1.2)
            ground = physics.bind(self._ground_geom)
            ground.friction[0] = random_sliding_friction

    def _randomize_current(
            self,
            physics: mjcf.Physics
            ) -> None:
        if self._random_current:
            angle = brb.brb_random_state.uniform(0, 2 * np.pi)
            direction = np.array([np.cos(angle), np.sin(angle), 0.0])
            strength = 0.1
            physics.model.opt.wind = strength * direction

            current_arrow = physics.bind(self._current_arrow)
            current_arrow.quat = euler2quat(np.pi / 2, np.pi / 2, np.pi / 2 + angle)

    def randomize(
            self,
            physics: mjcf.Physics,
            random_state: np.random.RandomState
            ) -> None:
        self._generate_random_height_and_light_maps(save_assets=True)
        self._update_ground_hfield(physics=physics)
        self._update_ground_texture(physics=physics)
        self._randomize_ground_friction(physics=physics)
        self._randomize_current(physics=physics)
        self._randomize_obstacles(physics=physics)

    def initialize_episode(
            self,
            physics: mjcf.Physics,
            random_state: np.random.RandomState
            ) -> None:
        self._previous_light_shift_time = 0

    def __del__(
            self
            ) -> None:
        dynamic_assets = [self._light_map_asset_path, self._heightmap_asset_path]
        for dynamic_asset_path in dynamic_assets:
            if Path(dynamic_asset_path).exists():
                os.remove(dynamic_asset_path)
