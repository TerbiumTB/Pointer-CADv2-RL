import cv2
import math
import numpy as np
from occwl.solid import Solid
from scipy.spatial.transform import Rotation as R



class CoordinateSystem(object):

    def __init__(self, rotation: R, origin) -> None:
        self.rotation = rotation
        self.origin = origin


    @property
    def normal(self):
        z_axis = self.rotation.as_matrix()[2]
        normal = z_axis / np.linalg.norm(z_axis)
        return normal


    @staticmethod
    def from_dict(stat):
        origin = np.array([stat["origin"]["x"], stat["origin"]["y"], stat["origin"]["z"]])
        x_axis_3d = np.array([stat["x_axis"]["x"], stat["x_axis"]["y"], stat["x_axis"]["z"]])
        y_axis_3d = np.array([stat["y_axis"]["x"], stat["y_axis"]["y"], stat["y_axis"]["z"]])
        z_axis_3d = np.array([stat["z_axis"]["x"], stat["z_axis"]["y"], stat["z_axis"]["z"]])
        rotation = R.from_matrix(np.vstack((x_axis_3d, y_axis_3d, z_axis_3d)))

        coord = CoordinateSystem(rotation, origin)

        return coord
    

    def world2relative(self, position):
        return self.rotation.apply(position - self.origin)


    def relative2world(self, position):
        position = np.asarray(position)
    
        if position.shape[-1] == 2:
            zeros = np.zeros_like(position[..., :1])
            position = np.concatenate([position, zeros], axis=-1)
        return self.rotation.inv().apply(position) + self.origin


    def transform(self, movement=(0,0,0), scale=1):
        self.origin = (self.origin + movement) * scale


    @staticmethod
    def __calculate_axis_intersections(point=None, normal=None):
        """
        计算平面与 x、y、z 轴的交点。
        
        参数：
        - point: (x0, y0, z0)，平面上一点
        - normal: (a, b, c)，平面的法向量
        
        返回：
        - (x_axis_intersect, y_axis_intersect, z_axis_intersect)
        三个值分别是与 x、y、z 轴的交点的非零值，如不存在交点则为 None。
        """
        x0, y0, z0 = point
        a, b, c = normal
        
        # 平面方程常数项 d
        d = -(a * x0 + b * y0 + c * z0)

        # 与 x 轴交点：y=0, z=0
        x_axis_intersect = None
        if a != 0:
            x_axis_intersect = -d / a
        
        # 与 y 轴交点：x=0, z=0
        y_axis_intersect = None
        if b != 0:
            y_axis_intersect = -d / b
        
        # 与 z 轴交点：x=0, y=0
        z_axis_intersect = None
        if c != 0:
            z_axis_intersect = -d / c
        
        return x_axis_intersect, y_axis_intersect, z_axis_intersect

    
    def __axis_intersections(self):
        return CoordinateSystem.__calculate_axis_intersections(self.origin, self.normal)

    def __calculate_rotation_angle(self, base, target):
        """
        计算从 base 向量逆时针旋转到 target 向量的角度。
        
        参数:
            base, target: numpy 数组，表示三维向量
        
        返回:
            逆时针旋转的角度（float）
        """
        # base = np.array(base, dtype=np.float64)
        # target = np.array(target, dtype=np.float64)
        base_unit = base / np.linalg.norm(base)
        target_unit = target / np.linalg.norm(target)

        # 点积求夹角（弧度）
        dot = np.clip(np.dot(base_unit, target_unit), -1.0, 1.0)
        angle = np.arccos(dot)

        # 判断方向：使用 (base × target) · normal
        cross = np.cross(base_unit, target_unit)
        direction = np.dot(cross, self.normal)

        if direction < 0:
            angle = 2 * np.pi - angle

        return np.degrees(angle)


    def to_vector(self, parameters=None):
        from misc import TOKEN, map_patameter_to_indices

        x0, y0, z0 = self.__axis_intersections()
        x, y, z = self.origin
        nx, ny, nz = self.normal
        
        # 创建一个带轴名称和索引的列表
        abs_components = [
            ('x', abs(nx), nx),
            ('y', abs(ny), ny),
            ('z', abs(nz), nz)
        ]
        
        # 按绝对值从大到小排序
        sorted_components = sorted(abs_components, key=lambda item: item[1], reverse=True)

        results = []

        for axis_name, abs_normal, normal in sorted_components:
            if normal == 0:
                continue

            if axis_name == 'x':
                base_vec = np.array((0, 1, 0)) if normal > 0 else np.array((0, 0, 1))
                if normal > 0 and y0 is not None:
                    base_vec = np.array((0, y0, 0)) - np.array((x0, 0, 0))
                    base_vec *= -1 if y0 < 0 else 1
                elif normal < 0 and z0 is not None:
                    base_vec = np.array((0, 0, z0)) - np.array((x0, 0, 0))
                    base_vec *= -1 if z0 < 0 else 1

                angle = self.__calculate_rotation_angle(base_vec, self.relative2world((1, 0, 0)) - self.origin)
                if normal > 0:
                    results.append((TOKEN.index("<|direction_x+|>"), y, z, angle))
                else:
                    results.append((TOKEN.index("<|direction_x-|>"), z, y, angle))
            elif axis_name == 'y':
                base_vec = np.array((0, 0, 1)) if normal > 0 else np.array((1, 0, 0))
                if normal > 0 and z0 is not None:
                    base_vec = np.array((0, 0, z0)) - np.array((0, y0, 0))
                    base_vec *= -1 if z0 < 0 else 1
                elif normal < 0 and x0 is not None:
                    base_vec = np.array((x0, 0, 0)) - np.array((0, y0, 0))
                    base_vec *= -1 if x0 < 0 else 1

                angle = self.__calculate_rotation_angle(base_vec, self.relative2world((1, 0, 0)) - self.origin)
                if normal > 0:
                    results.append((TOKEN.index("<|direction_y+|>"), z, x, angle))
                else:
                    results.append((TOKEN.index("<|direction_y-|>"), x, z, angle))
            elif axis_name == 'z':
                base_vec = np.array((1, 0, 0)) if normal > 0 else np.array((0, 1, 0))
                if normal > 0 and x0 is not None:
                    base_vec = np.array((x0, 0, 0)) - np.array((0, 0, z0))
                    base_vec *= -1 if x0 < 0 else 1
                elif normal < 0 and y0 is not None:
                    base_vec = np.array((0, y0, 0)) - np.array((0, 0, z0))
                    base_vec *= -1 if y0 < 0 else 1

                angle = self.__calculate_rotation_angle(base_vec, self.relative2world((1, 0, 0)) - self.origin)
                if normal > 0:
                    results.append((TOKEN.index("<|direction_z+|>"), x, y, angle))
                else:
                    results.append((TOKEN.index("<|direction_z-|>"), y, x, angle))
        
        # print(results)
        vectors = []
        for result in results:
            u = map_patameter_to_indices(result[1], parameters, "length")
            v = map_patameter_to_indices(result[2], parameters, "length")
            theta = map_patameter_to_indices((result[3] + 360) % 360, parameters, "angle")
            vec = [[result[0], None], [TOKEN.index("<|length_value|>"), u], [TOKEN.index("<|length_value|>"), v], [TOKEN.index("<|angle_value|>"), theta]]

            vectors.append(vec)

        return vectors
    

    @staticmethod
    def __calculate_plane_origin(x, y, z, normal: np.array, point: np.array):
        """
        x, y, z: float or None，坐标中未知的用 None 表示
        normal: np.array, 法向量 [A, B, C]
        point: np.array, 平面上一点 [x0, y0, z0]

        返回：
            计算后的 (x, y, z)
        """
        A, B, C = normal
        x0, y0, z0 = point

        known_coords = [c is not None for c in (x, y, z)]
        if known_coords.count(True) != 2:
            raise ValueError("必须恰好给出两个坐标，另一个坐标为 None")

        if x is None:
            if A == 0:
                raise ValueError("无法计算 x，因为 A = 0")
            x = float(x0 - (B * (y - y0) + C * (z - z0)) / A)

        elif y is None:
            if B == 0:
                raise ValueError("无法计算 y，因为 B = 0")
            y = float(y0 - (A * (x - x0) + C * (z - z0)) / B)

        elif z is None:
            if C == 0:
                raise ValueError("无法计算 z，因为 C = 0")
            z = float(z0 - (A * (x - x0) + B * (y - y0)) / C)

        return x, y, z


    @staticmethod
    def __rotate_vector_by_angle(base: np.ndarray, normal: np.ndarray, angle_deg: float) -> np.ndarray:
        """
        逆时针绕法向量 normal 旋转 base 向量指定角度（角度制）
        :param base: 3D向量，np.array([x, y, z])
        :param normal: 旋转轴向量（法向量），不必归一化
        :param angle_deg: 旋转角度，单位为度
        :return: 旋转后的向量 np.ndarray
        """
        angle_rad = np.deg2rad(angle_deg)  # 角度转弧度
        k = normal / np.linalg.norm(normal)  # 归一化旋转轴
        
        base_rot = (base * np.cos(angle_rad) +
                    np.cross(k, base) * np.sin(angle_rad) +
                    k * np.dot(k, base) * (1 - np.cos(angle_rad)))
        
        return base_rot


    @staticmethod
    def from_vector(vector, ref_normal, ref_origin, parameters=None, strict=True):
        from misc import TOKEN, map_indices_to_parameter

        dir_normal = {"<|direction_x+|>": np.array([1, 0, 0]), "<|direction_x-|>": np.array([-1, 0, 0]), 
                      "<|direction_y+|>": np.array([0, 1, 0]), "<|direction_y-|>": np.array([0, -1, 0]), 
                      "<|direction_z+|>": np.array([0, 0, 1]), "<|direction_z-|>": np.array([0, 0, -1])}
        dir_idx = [TOKEN.index(token_name) for token_name in dir_normal]

        DIR_positions = [i for i, x in enumerate(vector) if x[0] in dir_idx]
        if len(DIR_positions) == 0:
            raise ValueError("Not enough elements to decode Coordinate System from vector.")
        
        for d_idx in DIR_positions:
            u, v, theta = None, None, None
            for value_idx in range(d_idx + 1, len(vector)):
                if u is None and vector[value_idx][0] == TOKEN.index("<|length_value|>"):
                    u = map_indices_to_parameter(vector[value_idx][1], parameters, "length")
                elif v is None and vector[value_idx][0] == TOKEN.index("<|length_value|>"):
                    v = map_indices_to_parameter(vector[value_idx][1], parameters, "length")
                elif theta is None and vector[value_idx][0] == TOKEN.index("<|angle_value|>"):
                    theta = map_indices_to_parameter(vector[value_idx][1], parameters, "angle")
                else:
                    break

            direction = TOKEN[vector[d_idx][0]]
            dir_dot = np.dot(ref_normal, dir_normal[direction])

            if dir_dot == 0 or u is None or v is None or theta is None:
                if strict: raise ValueError("Not enough elements to decode Coordinate System from vector.")
                else: continue

            normal = ref_normal if dir_dot > 0 else -ref_normal
            x0, y0, z0 = CoordinateSystem.__calculate_axis_intersections(ref_origin, normal)
            origin = [None, None, None]
            rotation_ref = None
            if direction == "<|direction_x+|>":
                origin = [None, u, v]
                if y0 is not None:
                    rotation_ref = np.array((0, y0, 0)) - np.array((x0, 0, 0))
                    rotation_ref *= -1 if y0 < 0 else 1
                else:
                    rotation_ref = np.array((0, 1, 0))
            elif direction == "<|direction_x-|>":
                origin = [None, v, u]
                if z0 is not None:
                    rotation_ref = np.array((0, 0, z0)) - np.array((x0, 0, 0))
                    rotation_ref *= -1 if z0 < 0 else 1
                else:
                    rotation_ref = np.array((0, 0, 1))
            elif direction == "<|direction_y+|>":
                origin = [v, None, u]
                if z0 is not None:
                    rotation_ref = np.array((0, 0, z0)) - np.array((0, y0, 0))
                    rotation_ref *= -1 if z0 < 0 else 1
                else:
                    rotation_ref = np.array((0, 0, 1))
            elif direction == "<|direction_y-|>":
                origin = [u, None, v]
                if x0 is not None:
                    rotation_ref = np.array((x0, 0, 0)) - np.array((0, y0, 0))
                    rotation_ref *= -1 if x0 < 0 else 1
                else:
                    rotation_ref = np.array((1, 0, 0))
            elif direction == "<|direction_z+|>":
                origin = [u, v, None]
                if x0 is not None:
                    rotation_ref = np.array((x0, 0, 0)) - np.array((0, 0, z0))
                    rotation_ref *= -1 if x0 < 0 else 1
                else:
                    rotation_ref = np.array((1, 0, 0))
            elif direction == "<|direction_z-|>":
                origin = [v, u, None]
                if y0 is not None:
                    rotation_ref = np.array((0, y0, 0)) - np.array((0, 0, z0))
                    rotation_ref *= -1 if y0 < 0 else 1
                else:
                    rotation_ref = np.array((0, 1, 0))

            origin = CoordinateSystem.__calculate_plane_origin(*origin, normal=normal, point=ref_origin)
            x_axis_3d = CoordinateSystem.__rotate_vector_by_angle(rotation_ref, normal, theta)
            y_axis_3d = np.cross(normal, x_axis_3d)
            rotation = R.from_matrix(np.vstack((x_axis_3d, y_axis_3d, normal)))

            return CoordinateSystem(rotation, origin)
        
        raise ValueError("Not enough elements to decode Coordinate System from vector.")


    def _json(self):
        from misc import ROUND_JSON

        normal = self.normal

        plane = None
        rotation = [0, 0, 0]
        if np.allclose(normal, [0, 0, -1], atol=1e-6):
            plane = "Bottom"
        elif np.allclose(normal, [1, 0, 0], atol=1e-6):
            plane = "Right"
        elif np.allclose(normal, [-1, 0, 0], atol=1e-6):
            plane = "Left"
        elif np.allclose(normal, [0, -1, 0], atol=1e-6):
            plane = "Front"
        elif np.allclose(normal, [0, 1, 0], atol=1e-6):
            plane = "Back"
        else:
            plane = "Top"
            rotation = self.rotation.as_euler("zyx", degrees=True)
        
        json_data = {
            "reference_plane": plane,
            "rotation": [round(r, ROUND_JSON) for r in rotation],
            "position": [round(p, ROUND_JSON) for p in self.origin]
        }
        return json_data