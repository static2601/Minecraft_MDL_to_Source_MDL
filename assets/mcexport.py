import io
import json
import logging.handlers
import math
import os
import shutil
import struct
import subprocess
import sys
import argparse
import base64
import zipfile
import asyncio
from idlelib.configdialog import is_int
from logging import debug
from operator import indexOf
from typing import List, Tuple, Any, ItemsView

from PIL import Image
from PIL.GifImagePlugin import getdata
from PIL.ImageCms import Flags
from PIL.ImageOps import contain

#fmt = '[%(levelname)s] %(asctime)s:\t%(message)s'
#fmt = '[%(levelname)s]%(funcName)s-%(lineno)d:\t\t%(message)s'
fmt = '[%(levelname)s]:\t\t%(message)s'
formatter = logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")

stderr_handler = logging.StreamHandler(sys.stderr)
stderr_handler.setFormatter(formatter)

file_handler = logging.handlers.RotatingFileHandler(
    'mcexport.log', backupCount=5, encoding='utf-8')
file_handler.setFormatter(formatter)

logger = logging.getLogger('mcexport')
logger.addHandler(stderr_handler)
logger.addHandler(file_handler)

logger.setLevel(logging.DEBUG)


def runcmd(cmd: str) -> Tuple[bool, int, str]:
    try:
        popen_params = {
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "stdin": subprocess.DEVNULL
        }

        if os.name == "nt":
            popen_params["creationflags"] = 0x08000000

        proc = subprocess.Popen(cmd, **popen_params)
        stdout, stderr = proc.communicate()
        return True, proc.returncode, stdout
    except Exception as err:
        return False, -1, err


reflect_lut = [(i / 255.0) ** 2.2 for i in range(256)]


def compute_reflectivity(data: bytes, channels=4) -> Tuple[float, float, float]:
    if channels != 3 and channels != 4:
        logger.error('unsupported texture format')
        exit(1)

    sX = sY = sZ = 0.0
    for i in range(0, len(data), channels):
        sX += reflect_lut[data[i]]
        sY += reflect_lut[data[i + 1]]
        sZ += reflect_lut[data[i + 2]]

    inv = 1.0 / (len(data) / 4)
    return sX * inv, sY * inv, sZ * inv


def save_vtf(images: List[Image.Image], fp: str, no_refle=False):
    def i8(x):
        return struct.unpack('B', x.to_bytes(1, 'little'))

    def i16(x):
        return struct.unpack('BB', x.to_bytes(2, 'little'))

    def i32(x):
        return struct.unpack('BBBB', x.to_bytes(4, 'little'))

    def f32(x):
        return struct.unpack('BBBB', struct.pack('f', float(x)))

    if isinstance(images, Image.Image):
        images = [images]

    sX = sY = sZ = 0.0
    width, height = images[0].size
    mode = images[0].mode

    if mode == 'RGB':
        channels = 3
    elif mode == 'RGBA':
        channels = 4
    else:
        logger.error('unsupported texture format')
        exit(1)

    raws = []
    for im in images:
        if im.size != (width, height) or mode != im.mode:
            logger.error('Animated texture mismatch')
            exit(1)

        raw = im.tobytes()
        raws += [raw]
        if not no_refle:
            x, y, z = compute_reflectivity(raw, channels)
            sX += x
            sY += y
            sZ += z

    if no_refle:
        sX = sY = sZ = 1.0
    else:
        inv = 1 / len(raws)
        sX /= inv
        sY /= inv
        sZ /= inv

    os.makedirs(os.path.dirname(fp), exist_ok=True)
    vtf = open(fp, 'wb')

    # common vtf header, define v7.2
    vtf.write(bytes([
        0x56, 0x54, 0x46, 0x00, 0x07, 0x00, 0x00, 0x00,
        0x02, 0x00, 0x00, 0x00, 0x50, 0x00, 0x00, 0x00,
    ]))

    # image size,
    # flag: Point Sampling | No Mipmaps | No Level Of Detail | Eight Bit Alpha
    # number of frames (1 for no animation), first frame of animation
    vtf.write(bytes([
        *i16(width), *i16(height), 0x01, 0x23, 0x00, 0x00,
        *i16(len(raws)), *i16(0), 0x00, 0x00, 0x00, 0x00,
        *f32(sX), *f32(sY),
        *f32(sZ), 0x00, 0x00, 0x00, 0x00,
    ]))

    # bump scale (1.0f), image format: IMAGE_FORMAT_RGBA8888(0) or IMAGE_FORMAT_RGB888(2)
    # mipcount, no low res image, depth, for 7.2+
    vtf.write(bytes([
        0x00, 0x00, 0x80, 0x3F, *i32(0 if mode == 'RGBA' else 2),
        0x01, 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x01,
        0x00
    ]))

    # padding
    vtf.write(bytes(80 - vtf.tell()))

    # image data
    for raw in raws:
        vtf.write(raw)

    vtf.close()


QC_MODELBASE = """
$cdmaterials "{path}"
$ambientboost
$scale {pixel_scale}
"""

QC_HEADER = """// Template header
$definevariable mdlname "{model_file}"
$include "modelbase_1.qci"
{anim_sequences}
"""

QC_FOOTER = """
// Block Template
$surfaceprop "default"
$keyvalues
{
    prop_data
    {
        "base" "Plastic.Medium"
    }
}

$modelname $mdlname$.mdl

$model "Body" $mesh$
$applytextures // Call macro

$sequence idle $mesh$.smd loop fps 1.00
$collisionmodel $mesh$.smd
{
	$concave
	$mass 50.0
}
"""

VMT_MODELS_TEMPLATE = """VertexLitGeneric
{{
    "$basetexture" "{cdmaterials}/{texture_file}"
    "$surfaceprop" "{surfaceprop}"
    "$alphatest" "1"
    {proxy}
}}
"""

VMT_PROXY_ANIMATION = """
Proxies
{{
    AnimatedTexture
    {{
        animatedtexturevar $basetexture
        animatedtextureframenumvar $frame
        animatedtextureframerate {frametime}
    }}
}}
"""

missing_texture = 'iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAAIGNIUk0AAHolAACAgwAA+f8AAIDpAAB1MAAA6mAAADqYAAAXb5JfxUYAAAAjSURBVCjPY/zD8J8BG2BhYMQqzjiqgSYaGHAAXAaNaqCJBgBNyh/pMWe+mgAAAABJRU5ErkJggg=='


class LineBuilder:
    def __init__(self):
        self.str = ''

    def __enter__(self):
        return self

    def __exit__(self, type, value, trace):
        pass

    def __call__(self, text='', end='\n'):
        self.str += text + end

    def __iadd__(self, text):
        self.str += text
        return self

    def __str__(self):
        return self.str

    def __len__(self):
        return len(self.str)


class Vector:
    def __init__(self, x=0, y=0, z=0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def dot(self, other) -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def __abs__(self):
        return math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

    def __pos__(self):
        return Vector(self.x, self.y, self.z)

    def __neg__(self):
        return Vector(-self.x, -self.y, -self.z)

    def __eq__(self, other):
        return abs(self - other) <= 1e-6

    def __ne__(self, other):
        return abs(self - other) > 1e-6

    def __add__(self, other):
        if isinstance(other, Vector):
            return Vector(self.x + other.x, self.y + other.y, self.z + other.z)
        else:
            return Vector(self.x + other, self.y + other, self.z + other)

    def __sub__(self, other):
        if isinstance(other, Vector):
            return Vector(self.x - other.x, self.y - other.y, self.z - other.z)
        else:
            return Vector(self.x - other, self.y - other, self.z - other)

    def __mul__(self, other):
        if isinstance(other, Vector):
            return Vector(self.x * other.x, self.y * other.y, self.z * other.z)
        else:
            return Vector(self.x * other, self.y * other, self.z * other)

    def __div__(self, other):
        if isinstance(other, Vector):
            return Vector(self.x / other.x, self.y / other.y, self.z / other.z)
        else:
            return Vector(self.x / other, self.y / other, self.z / other)

    def __truediv__(self, other):
        if isinstance(other, Vector):
            return Vector(self.x / other.x, self.y / other.y, self.z / other.z)
        else:
            return Vector(self.x / other, self.y / other, self.z / other)

    def __radd__(self, other):
        if isinstance(other, Vector):
            return Vector(other.x + self.x, other.y + self.y, other.z + self.z)
        else:
            return Vector(other + self.x, other + self.y, other + self.z)

    def __rsub__(self, other):
        if isinstance(other, Vector):
            return Vector(other.x - self.x, other.y - self.y, other.z - self.z)
        else:
            return Vector(other - self.x, other - self.y, other - self.z)

    def __rmul__(self, other):
        if isinstance(other, Vector):
            return Vector(other.x * self.x, other.y * self.y, other.z * self.z)
        else:
            return Vector(other * self.x, other * self.y, other * self.z)

    def __rdiv__(self, other):
        if isinstance(other, Vector):
            return Vector(other.x / self.x, other.y / self.y, other.z / self.z)
        else:
            return Vector(other / self.x, other / self.y, other / self.z)

    def __rtruediv__(self, other):
        if isinstance(other, Vector):
            return Vector(other.x / self.x, other.y / self.y, other.z / self.z)
        else:
            return Vector(other / self.x, other / self.y, other / self.z)

    def __iadd__(self, other):
        if isinstance(other, Vector):
            self.x += other.x
            self.y += other.y
            self.z += other.z
        else:
            self.x += other
            self.y += other
            self.z += other
        return self

    def __isub__(self, other):
        if isinstance(other, Vector):
            self.x -= other.x
            self.y -= other.y
            self.z -= other.z
        else:
            self.x -= other
            self.y -= other
            self.z -= other
        return self

    def __imul__(self, other):
        if isinstance(other, Vector):
            self.x *= other.x
            self.y *= other.y
            self.z *= other.z
        else:
            self.x *= other
            self.y *= other
            self.z *= other
        return self

    def __idiv__(self, other):
        if isinstance(other, Vector):
            self.x /= other.x
            self.y /= other.y
            self.z /= other.z
        else:
            self.x /= other
            self.y /= other
            self.z /= other
        return self

    def __itruediv__(self, other):
        if isinstance(other, Vector):
            self.x /= other.x
            self.y /= other.y
            self.z /= other.z
        else:
            self.x /= other
            self.y /= other
            self.z /= other
        return self

    def __getitem__(self, key):
        if not isinstance(key, int):
            raise KeyError(f'key: {key} must be interger')

        if key == 0:
            return self.x
        elif key == 1:
            return self.y
        elif key == 2:
            return self.z
        else:
            raise KeyError(f'key: {key} dose not exists')

    def __setitem__(self, key, val):
        if not isinstance(key, int):
            raise KeyError(f'key: {key} must be interger')

        if key == 0:
            self.x = val
        elif key == 1:
            self.y = val
        elif key == 2:
            self.z = val
        else:
            raise KeyError(f'key: {key} dose not exists')

    def __str__(self):
        return f'{self.x:.5f} {self.y:.5f} {self.z:.5f}'

    def __repr__(self):
        return f'{self.x:.1f} {self.y:.1f} {self.z:.1f}'

    def rotate_x(self, angle):
        rad = math.radians(angle)
        cos = math.cos(rad)
        sin = math.sin(rad)
        return Vector(self.x, cos * self.y - sin * self.z, sin * self.y + cos * self.z)

    def rotate_y(self, angle):
        rad = math.radians(angle)
        cos = math.cos(rad)
        sin = math.sin(rad)
        return Vector(cos * self.x + sin * self.z, self.y, -sin * self.x + cos * self.z)

    def rotate_z(self, angle):
        rad = math.radians(angle)
        cos = math.cos(rad)
        sin = math.sin(rad)
        return Vector(cos * self.x - sin * self.y, sin * self.x + cos * self.y, self.z)

    def rotate(self, angle, center=None):
        vec = Vector(self.x, self.y, self.z)
        if angle[0] != 0:
            if center is not None:
                vec -= center
            vec = vec.rotate_x(angle[0])
            if center is not None:
                vec += center

        if angle[1] != 0:
            if center is not None:
                vec -= center
            vec = vec.rotate_y(angle[1])
            if center is not None:
                vec += center

        if angle[2] != 0:
            if center is not None:
                vec -= center
            vec = vec.rotate_z(angle[2])
            if center is not None:
                vec += center

        return vec


class Vertex:
    def __init__(self, pos: Vector, uv: Vector):
        self.pos = pos
        self.uv = uv

    def translate(self, vec: Vector):
        self.pos += vec

    def scale(self, scale: Vector):
        self.pos *= scale

    def rotate(self, angle: Vector, center=None):
        self.pos = self.pos.rotate(angle, center)

    def to_smd(self, normal: Vector, link: int, bone: int) -> str:
        # <int|Parent bone> <float|PosX PosY PosZ> <normal|NormX NormY NormZ> <normal|U V> [ignores]
        return f'0  {self.pos}  {normal}  {self.uv.x:.5f} {self.uv.y:.5f}  {link} {bone} 1'


class Face:
    def __init__(self, texture: str, vertices: List[Vertex], norm: Vector, link: int, bone: int):
        self.tex = texture
        self.vert = vertices
        self.norm = norm
        self.link = link
        self.bone = bone

    def translate(self, vec: Vector):
        for v in self.vert:
            v.translate(vec)

    def scale(self, scale: Vector):
        for v in self.vert:
            v.scale(scale)

    def rotate(self, angle: Vector, center=None):
        for v in self.vert:
            v.rotate(angle, center)

    def to_smd(self) -> str:
        vert = self.vert
        builder = self.tex + '\n'
        builder += vert[0].to_smd(self.norm, self.link, self.bone) + '\n'
        builder += vert[1].to_smd(self.norm, self.link, self.bone) + '\n'
        builder += vert[2].to_smd(self.norm, self.link, self.bone) + '\n'
        builder += self.tex + '\n'
        builder += vert[1].to_smd(self.norm, self.link, self.bone) + '\n'
        builder += vert[3].to_smd(self.norm, self.link, self.bone) + '\n'
        builder += vert[2].to_smd(self.norm, self.link, self.bone) + '\n'
        return builder


class Cube:
    def __init__(self, position: Vector, size: Vector):
        self.size = size
        self.position = position
        #self.link = link
        #self.bone = bone
        self.faces = []

    def add_face(self, face: int, texture: str, uv: List[Vector], link: int, bone: int):
        if face == 0:  # east
            norm = Vector(-1.0, 0.0, 0.0)
            vertices = [
                Vertex(Vector(0.0, 1.0, 1.0), uv[0]),
                Vertex(Vector(0.0, 1.0, 0.0), uv[1]),
                Vertex(Vector(0.0, 0.0, 1.0), uv[2]),
                Vertex(Vector(0.0, 0.0, 0.0), uv[3]),
            ]
        elif face == 1:  # west
            norm = Vector(1.0, 0.0, 0.0)
            vertices = [
                Vertex(Vector(1.0, 0.0, 1.0), uv[0]),
                Vertex(Vector(1.0, 0.0, 0.0), uv[1]),
                Vertex(Vector(1.0, 1.0, 1.0), uv[2]),
                Vertex(Vector(1.0, 1.0, 0.0), uv[3]),
            ]

        elif face == 2:  # up
            norm = Vector(0.0, 0.0, 1.0)
            vertices = [
                Vertex(Vector(0.0, 1.0, 1.0), uv[0]),
                Vertex(Vector(0.0, 0.0, 1.0), uv[1]),
                Vertex(Vector(1.0, 1.0, 1.0), uv[2]),
                Vertex(Vector(1.0, 0.0, 1.0), uv[3]),
            ]

        elif face == 3:  # down
            norm = Vector(0.0, 0.0, -1.0)
            vertices = [
                Vertex(Vector(0.0, 0.0, 0.0), uv[0]),
                Vertex(Vector(0.0, 1.0, 0.0), uv[1]),
                Vertex(Vector(1.0, 0.0, 0.0), uv[2]),
                Vertex(Vector(1.0, 1.0, 0.0), uv[3]),
            ]

        elif face == 4:  # south
            norm = Vector(0.0, 1.0, 0.0)
            vertices = [
                Vertex(Vector(1.0, 1.0, 1.0), uv[0]),
                Vertex(Vector(1.0, 1.0, 0.0), uv[1]),
                Vertex(Vector(0.0, 1.0, 1.0), uv[2]),
                Vertex(Vector(0.0, 1.0, 0.0), uv[3]),
            ]

        elif face == 5:  # north
            norm = Vector(0.0, -1.0, 0.0)
            vertices = [
                Vertex(Vector(0.0, 0.0, 1.0), uv[0]),
                Vertex(Vector(0.0, 0.0, 0.0), uv[1]),
                Vertex(Vector(1.0, 0.0, 1.0), uv[2]),
                Vertex(Vector(1.0, 0.0, 0.0), uv[3]),
            ]

        # reference cube is center @ [0.5, 0.5, 0.5]
        face = Face(texture, vertices, norm, link, bone)
        face.translate(Vector(-0.5, -0.5, -0.5))

        face.scale(self.size)
        face.translate(self.position)

        self.faces += [face]

    def translate(self, vec: Vector):
        for face in self.faces:
            face.translate(vec)

    def scale(self, scale):
        for face in self.faces:
            face.scale(scale)

    def rotate(self, angle: Vector, center=None, rescale=False):
        if center is not None:
            self.translate(-center)

        for face in self.faces:
            face.rotate(angle)

        if rescale:
            scaled = self.position.rotate(angle)
            scale = self.position.dot(
                self.position) / self.position.dot(scaled)  # porjection
            self.scale(Vector(scale, scale, 1))

        if center is not None:
            self.translate(center)

    def to_smd(self) -> str:
        builder = ''
        for face in self.faces:
            builder += face.to_smd()
        return builder

    def to_smdbones(self, center: Vector) -> str:
        return f'{center}  0.00000 0.00000 0.00000'


class SMDModel:
    def __init__(self):
        self.textures = {}
        self.cubes = []

        self.min = Vector()
        self.max = Vector()

    def add_cube(self, position: Vector, scale: Vector) -> Cube:
        cube = Cube(position, scale)
        self.cubes += [cube]
        return cube

    def translate(self, vec: Vector):
        for cube in self.cubes:
            cube.translate(vec)

    def to_smd(self) -> str:
        builder = ''
        for e in self.cubes:
            builder += e.to_smd()
        return builder

    def to_smdbones(self) -> str:
        builder = ''
        for i, e in enumerate(self.cubes):
            # assume it center at 0
            builder += f'{i} {e.to_smdbones(Vector())}\n'
        return builder


class Bone:
    def __init__(self, bone: dict):
        self.bone_dict = bone
        self.bone_dict: dict = {}
        self.name: str = ''
        self.parent: str = ''
        self.origin: list = []
        self.rotation: list = []
        self.cubes: list = []
        self.cubes_str_list: list[str] = []
        self.cubes_list: list = []
        self.cubes_dict: list[dict] = []
        #for bone in bones:
        if 'name' in bone:
            self.name = bone['name']
        if 'pivot' in bone:
            self.origin = bone['pivot']
        if 'rotation' in bone:
            self.rotation = bone['rotation']
        if 'parent' in bone:
            self.parent = bone['parent']
        else:
            self.parent = bone['name']
        if 'cubes' in bone:
            cubes = bone['cubes']
            #self.cubes = bone['cubes']
            for cube in cubes:
                self.cubes.append(cube)
            #self.set_cube_str_list()


    # def add_cube_str_list(self):
    #     self.cubes_str_list.append(self.name)

    def get_cubes_str_list(self):
        # if not self.cubes_list:
        #     return self.cubes_list
        # if not self.cubes_dict:
        #     return self.cubes_dict
        # return None
        return self.cubes_str_list

    def get_bone(self) -> dict:
        #TODO have both ways of getting bones: of geometry json and other being by converted json
        # because if the model parts need adjusted in converted json, bones wont match if from geometry json
        new_dict: dict = {'name': self.name, 'parent': self.parent,'origin': self.origin, 'rotation': self.rotation, 'cubes': self.cubes_list}
        #TODO
        #if self.name == self.parent :
        #     new_dict.update({'cubes': self.get_cubes_str_list()})
        return new_dict

    # def set_cubes(self, cubes: Any):
    #     if type(cubes) is list:
    #         self.cubes_list = cubes
    #     elif type(cubes) is dict:
    #         self.cubes_dict = cubes
    #     else:
    #         logger.warning(f"Trying to set Bone.cubes not of a list or dict type.")

    def add_cube(self, cube: Any):
        if type(cube) is not Bone:
            self.cubes_list.append(cube.name)
        else:
            logger.error(f"Error: cube must be a Bone type.")

    def add_cube_as_int(self, cube: int):
        self.cubes_list.append(cube)


class GeoModel:
    def __init__(self, entity_path: str):
        self.entity_path = entity_path
        self.bones: list[Bone] = []
        self.bones_list: list = []
        self.entity_id: str = ''
        self.entity_tex: str = ''
        self.entity_geo: str = ''
        self.entity_anims: dict = {}
        self.entity_scripts: list = []
        self.entity_anim_path: str = ''
        self.geo_model_path: str = ''
        self.get_data(entity_path)

    def get_data(self, entity_path: str):
        if not os.path.exists(entity_path):
            logger.error(f"Error: entity_path does not exist. entity_path: {entity_path}")
            exit(1)

        with open(entity_path) as ff:
            logger.debug("entity_path exists, entity_json = json.load(f)")
            entity_json = json.load(ff)

            description = entity_json['minecraft:client_entity']['description']
            self.entity_id = description['identifier']
            self.entity_tex = description['materials']['default']
            self.entity_geo = description['geometry']['default']
            self.entity_anims = description['animations']
            self.entity_scripts = description['scripts']['animate']

            anim_dir: str = os.path.join(entity_path.split('entity')[0], "animations")
            self.entity_anim_path = find_anim_in_dir(anim_dir, self.entity_anims)
            self.get_bones()

    def get_bones(self):
        # full path to geo_model in models directory
        geo_model_path: str = self.find_geo_in_dir(self.entity_geo)
        self.geo_model_path = geo_model_path
        geo_jmodel: Any
        description: Any
        bones: list = []
        #cubes: list = []

        with open(geo_model_path, 'r') as jm:
            geo_jmodel = json.load(jm)
            if 'minecraft:geometry' in geo_jmodel:
                logger.debug(f"\t[JSON_Path]:{geo_model_path}")
                for geo in geo_jmodel['minecraft:geometry']:
                    if 'description' in geo:
                        description = geo['description']
                    if 'bones' in geo:
                        for bone in geo['bones']:
                            bones.append(bone)
                        # for bne in bones:
                        #     #bone = bone
                        #     if 'cubes' in bne:
                        #         #cubes = cubes['cubes']
                        #         for cube in bne['cubes']:
                        #             pass
                        #logger.debug(f"bones from json: {bones}")
            else:
                logger.error(f"'minecraft:geometry' not found in file path: '{geo_model_path}'")
                return None

        if not bones:
            logger.error(f"bones list is None.")
            return None

        cube_count: int = 0
        # b_chest = None
        # b_lid = None
        for b in bones:

            bone: Bone = Bone(b)
            logger.debug(f"bone.name: {bone.name}")
            # if not b_chest and bone.name == 'chest':
            #     b_chest = bone
            # if not b_lid and bone.name == 'lid':
            #     b_lid = bone

            logger.debug(f"bone.cubes: {bone.cubes}")
            for cube in bone.cubes:
                logger.debug(f"cube_count: {cube_count}")
                logger.debug(f"cube: {cube}")
                #bone.cubes_list.append(cube_count)
                bone.add_cube_as_int(cube_count)
                cube_count += 1
                logger.debug(f"bone.cubes_list: {bone.cubes_list}")
                # if b_chest:
                #     logger.debug(f"b_chest: {b_chest.name}")
                # if b_lid:
                #     logger.debug(f"b_lid: {b_lid.name}")

            self.bones.append(bone)

        #TODO fix, not working, wont return cubes list with 'get_bone_dict'. ignoring for now

        # if parent bone added with no cubes, add them
        # parents: list = []
        # childs: list = []
        # #logger.debug(f"self.bones: {self.bones}")
        # for b in self.bones:
        #     bone: Bone = b
        #     #logger.debug(f"b: {b}")
        #     #logger.debug(f"bone.parent: { bone.parent}, bone.name: {bone.name}")
        #     if bone.parent == bone.name:
        #         parents.append(bone)
        #     else:
        #         childs.append(bone)
        # for b in parents:
        #     bone: Bone = b
        #     for c in childs:
        #         c_bone: Bone = c
        #         if c_bone.parent == bone:
        #             bone.add_cube(c_bone)
        #             logger.debug(f"adding cube to bone")
        # logger.debug(f"self.bones: {self.bones}")
        return self.bones

    def get_bones_dict(self) -> dict:
        #bones: list[Bone] = self.get_bones()
        bones_dict: dict = {}
        for bone in self.bones:
            bones_dict[bone.name] = bone.get_bone()
        return bones_dict

    def find_geo_in_dir(self, geo_model: str) -> Any:
        models_dir: str = os.path.join(self.entity_path.split('entity')[0], "models")
        for dirpath, _, file_names in os.walk(models_dir):
            for file_name in file_names:
                full_file_path = os.path.join(dirpath, file_name)
                if full_file_path.endswith('geo.json'):
                    try:
                        with open(full_file_path, 'r') as f:
                            if geo_model in f.read():
                                return full_file_path
                    except IOError as e:
                        logger.error(f"Error reading file {full_file_path}: {e}")
        return None


def export_texture(texture: str, texture_dir: str, out_dirs: str) -> None:
    #global im
    logger.debug("")
    logger.debug(f'export_texture(texture, texture_dir, out_dir)')
    logger.debug(f"\t-texture: {texture}")
    logger.debug(f"\t-texture_dir: {texture_dir}")
    logger.debug(f"\t-out_dirs: {out_dirs}")

    source: str = ""
    #im = Image.open(io.BytesIO(base64.b64decode(missing_texture)))
    out = os.path.normpath(os.path.join(out_dirs, texture))
    if texture == '$missing':
        im = Image.open(io.BytesIO(base64.b64decode(missing_texture)))
    else:
        source = os.path.normpath(os.path.join(texture_dir, texture + '.png'))
        if not os.path.exists(source):
            #TODO check shouldnt be needed, being done before this function
            logger.warning(f'\tTexture file does not exist: {source}, falling back to minecraft assets...')
            # if not exists, fall back to minecraft assets for textures
            texture_dir = os.path.join(mod_assets_path, "minecraft", "textures", "block")
            source = os.path.normpath(os.path.join(texture_dir, texture + '.png'))
            #im = Image.open(source)
            if not os.path.exists(source):
                #open_texture_file()
                logger.warning(f'\tTexture file still does not exist: {source}, skipping...')
                return

    im = Image.open(source)
    if im.format != 'PNG':
        logger.warning(f'\tSource texture is not PNG: {texture}')

    if im.mode != 'RGB' and im.mode != 'RGBA':
        im = im.convert('RGBA')

    meta = os.path.normpath(os.path.join(texture_dir, texture + '.png.mcmeta'))
    if os.path.exists(meta):
        logger.debug(f"\tpng.meta: {meta} exists, opening...")
        with open(meta) as f:
            animation = json.load(f)['animation']

        # TODO:
        if 'frames' in animation:
            logger.warning(
                f'  texture {texture} has "animation.frames", but is currently not supported')

        # minecraft animation 20 fps, it might cause some lag in csgo
        if 'frametime' in animation:
            frametime = 20 / animation['frametime']
        else:
            frametime = 20

        # slow it down or fps will drop when player up close
        frametime /= 2

        frame_size = im.width
        assert im.height % frame_size == 0, f'  invalid animated texture: {texture}'

        proxy = VMT_PROXY_ANIMATION.format(frametime=frametime)
        textureDimens[texture] = Vector(frame_size, frame_size)

        save_vtf([
            im.crop((0, i * frame_size, frame_size, i * frame_size + frame_size))
            for i in range(0, im.height // frame_size)
        ], out + '.vtf')
    else:
        proxy = ''
        textureDimens[texture] = Vector(*im.size)
        save_vtf(im, out + '.vtf')
        logger.debug(f"\tmeta: {meta} Does not exist")

    logger.debug("\tMaking VTF/VMT...")
    cdmaterials = os.path.join("models", mod_name)
    if sb_dir != "":
        cdmaterials = os.path.join("models", mod_name, sb_dir)

    logger.debug(f"\tcdmaterials: '{cdmaterials}'")
    with open(out + '.vmt', 'w') as f:
        logger.debug("  Writing VMT...")
        f.write(VMT_MODELS_TEMPLATE.format(
            cdmaterials=cdmaterials, texture_file=texture, surfaceprop='', proxy=proxy))

    game_material = os.path.join(game_path, 'materials', cdmaterials, texture)
    logger.debug(f"\tgame_path: '{game_path}'")
    logger.debug(f"\tgame_material: '{game_material}'")
    logger.debug(f"\tgame material path: {os.path.dirname(game_material)}")

    os.makedirs(os.path.dirname(game_material), exist_ok=True)
    shutil.copyfile(out + '.vmt', game_material + '.vmt')
    shutil.copyfile(out + '.vtf', game_material + '.vtf')

    logger.debug("<---end of exporting textures <--")
    logger.debug("")


def resolve_uv(texture: str) -> str:
    tex_file = resolve_texture(texture)
    if tex_file not in textureDimens:
        export_texture(tex_file, textures_path, out_textures_path)
    return tex_file


def resolve_texture(texture: str) -> str:
    if textureVars[texture][0] == '#':
        return resolve_texture(textureVars[texture][1:])
    return textureVars[texture]


def convert_uv(mcuv: List[float], rotation=0) -> List[Vector]:
    mcuv = [v / 16 for v in mcuv]
    ref = [
        (mcuv[0], 1 - mcuv[3]),
        (mcuv[0], 1 - mcuv[1]),
        (mcuv[2], 1 - mcuv[3]),
        (mcuv[2], 1 - mcuv[1]),
    ]

    # source uv coordinate
    uv = [
        Vector(ref[1][0], ref[1][1]),
        Vector(ref[0][0], ref[0][1]),
        Vector(ref[3][0], ref[3][1]),
        Vector(ref[2][0], ref[2][1]),
    ]

    for i in range(rotation // 90):
        uv = [uv[1], uv[3], uv[0], uv[2]]

    return uv


def open_texture_file(modname: str, texture_path: str, texture_val: str, new_jar_path: str, db: bool = True) -> bool:
    '''
    check if given texture path can be opened. else, find and extract so it can be run with 'export_textures'
    :param modname: mod name as in registery name eg: 'minecraft' in 'minecraft:block'
    :param texture_path: path to textures of model
    :param texture_val: name of the model
    :param new_jar_path: jar file to try, empty by default
    :param db: verbose debug
    :return: true if can be opened, false otherwise
    '''
    if db:
        logger.debug("")
        logger.debug(f"open_texture_file(modname, texture_path, texture_val, new_jar_path='')")
        logger.debug(f"\t-modname: '{modname}'")
        logger.debug(f"\t-texture_path: '{texture_path}'")
        logger.debug(f"\t-texture_val: '{texture_val}'")
        logger.debug(f"\t-new_jar_path: '{new_jar_path}'")

    t_path: str
    if is_geo:
        t_path = os.path.join(texture_path, ent_tex + '.png')
    else:
        t_path = os.path.join(texture_path, texture_val + '.png')
    if db: logger.debug(f"\tformed texture path, path: {t_path}")

    # if texture.png already extracted, path will be found
    # and not extract mcmeta data if trying again
    if os.path.exists(t_path):
        if db: logger.debug(f"\ttexture found at path: {t_path}")
        return True
    elif is_geo:
        logger.debug(f"Error: geometry texture_path not found. texture_path: '{t_path}'")
        exit(1)
    else:
        if db: logger.debug(f"\ttexture does not exist at path: {t_path}")
        if db: logger.debug(f"\tattempting texture extraction...")

        # assets path is base project_dir/assets
        # path is full path to file with '.json'
        # need suffixed with assets/model/block etc
        mod: str = modname
        # will not come in as minecraft unless texture has minecraft: prefix
        # so for below, if modname/texture not found, try minecraft/texture
        jar_path: str
        if db: logger.debug("\tchecking if new_jar_path defined...")
        if new_jar_path != "":
            if db: logger.debug(f"\tsetting jar_path to new_jar_path: {new_jar_path}")
            jar_path = new_jar_path
        else:
            #jar_path = mod_jar_path
            if db: logger.debug("\tnew_jar_path not set.")
            if mod == "minecraft":
                jar_path = mcjar_path
            else:
                jar_path = mod_jar_path

        if db: logger.debug(f"\tmod: {mod}")
        if db: logger.debug(f"\tjar_path: {jar_path}")
        # if we reran open_texture_file() with a new texture_path excluding block directory
        # only need to replace mod
        # TODO only replace mod, replace texture_prefix with texture_path

        texture_prefix: str
        # 'texture_path' always end in either block or textures
        # with sub_dirs applied to 'val'
        if texture_path.endswith("block"):
            texture_prefix = os.path.join("assets", mod, "textures", "block")
        else:
            texture_prefix = os.path.join("assets", mod, "textures")

        file_to_extract: str = os.path.join(texture_prefix, texture_val + '.png').replace("\\", "/")
        #logger.debug(f"\tfile_to_extract: {file_to_extract}")
        #logger.debug(f"\tpath: {path}")
        #logger.debug(f"\tassets_path: {assets_path}")

        # if cant extract file, needs to try as minecraft mod as textures
        # but why is it running minecraft as mod? should this be outside this function?
        # return false and try again
        if not extract_jar_file(jar_path, file_to_extract, assets_path, db):
            return False
            # #logger.debug(f"\t'file_to_extract' not found in 'jar_path'")
            # #TODO should not run multiple times, needs to run only once per file looking for...
            # # open_texture_file runs multiple times per file...
            # # maybe this part can be placed outside this scope and insert into new_jar_path before running...
            #
            # logger.debug(f"\tchecking minecraft .jar for file...")
            # # try with minecraft texture instead of current mod
            # jar_path = mcjar_path
            # mod = "minecraft"
            # texture_prefix = os.path.join("assets", mod, "textures", "block")
            # file_to_extract = os.path.join(texture_prefix, texture_val + '.png').replace("\\", "/")
            #
            # if not extract_jar_file(jar_path, file_to_extract, assets_path):
            #     logger.warning("\textracting texture as minecraft fallback returned false.")
            #     return False
            # else:
            #     #TODO should this try extracting png.meta?
            #     #logger.debug(f"\tfile extracted.")
            #     logger.debug(f"\tchecking if a png.meta file exists...")
            #     file_to_extract = os.path.join(texture_prefix, texture_val + '.png.mcmeta').replace("\\", "/")
            #     extract_jar_file(jar_path, file_to_extract, assets_path)
            #     return True

        else:
            # try extracting .png.mcmeta animations file
            #logger.debug(f"\tfile extracted successfully.")
            if db: logger.debug(f"\tchecking if a png.meta file exists...")
            file_to_extract = os.path.join(texture_prefix, texture_val + '.png.mcmeta').replace("\\", "/")
            extract_jar_file(jar_path, file_to_extract, assets_path, db)
            return True


def open_model_file(path: str, new_jar_path: str) -> Any:
    logger.debug("")
    logger.debug("open_model_file(path, new_jar_path)")
    logger.debug(f"\t-path: '{path}'")
    logger.debug(f"\t-new_jar_path: '{new_jar_path}'")

    # if texture is eg: "countertop": "block/polished_andesite"
    # try mod textures directory first, if that fails, try minecraft textures directory

    #logger.debug(f"\tOpening model file, path: {path}")
    if os.path.exists(path):
        with open(path) as ff:
            # TODO check if file is empty and reacquire
            logger.debug("\tModel file exists, returning json.load(f)")
            return json.load(ff)
    elif is_geo:
        logger.error(f"geo model file does not exist at path. path: '{path}'")
        exit(1)
    else:
        #return None
        logger.debug("\t\tModel file does not exist, trying to extract...")
        # assets path is base project_dir/assets
        # path is full path to file with '.json'
        # need suffixed with assets/model/block etc
        jar_path: str
        mod: str
        new_assets_path: str = path.split("assets\\assets\\")[1]
        if "minecraft" in new_assets_path:
            mod = "minecraft"
            jar_path = mcjar_path
            # if jar_path == "":
            #     logger.error("\t\tError: mc_jar path not set. Exiting.")
            #     exit(1)
        else:
            mod = mod_name
            #jar_path = getJarPath()
            logger.debug(f"\t\tchecking if new_jar_path set.")
            if new_jar_path != "":
                jar_path = new_jar_path
                logger.debug(f"\t\tjar_path set as new_jar_path. jar_path: {jar_path}")
            else:
                jar_path = mod_jar_path
                logger.debug(f"\t\tjar_path set as mod_jar_path. jar_path: {jar_path}")

        # split will return model name and sub directories after block/
        new_model_name: str = path.split("models\\block\\")[1]
        model_prefix: str = os.path.join("assets", mod, "models", "block")

        file_to_extract: str = os.path.join(model_prefix, new_model_name).replace("\\", "/")
        extract_jar_file(jar_path, file_to_extract, assets_path)

        if os.path.exists(path):
            with open(path) as ff:
                logger.debug("\t\treturning json.load(f)")
                return json.load(ff)
        else:
            logger.warning(f"\t\tTried extraction, but failed to find extracted file.")
            return None


def getJarPath() -> Any:
    logger.debug("")
    logger.debug("getJarPath()")
    # TODO getting from MOD_JARS will NOT work if not from sourcecraft_import
    #path: str = os.path.join(mod_assets_path, MOD_JARS)
    path: str = os.path.join(mod_assets_path, mod_jar_path)
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
            return data[mod_name]


def extract_jar_file(jar_file_path, file_to_extract, destination_dir, db: bool = True) -> bool:
    if db:
        logger.debug("")
        logger.debug(f"extract_jar_file(jar_file_path, file_to_extract, destination_dir)")
        logger.debug(f"\t-jar_file_path:   '{jar_file_path}'")
        logger.debug(f"\t-file_to_extract: '{file_to_extract}'")
        logger.debug(f"\t-destination_dir: '{destination_dir}'")

    if check_in_jar(jar_file_path, file_to_extract, db):
        #logger.debug("Found in JAR.")
        # Ensure the destination directory exists
        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)
        # Open the JAR file in read mode
        with zipfile.ZipFile(jar_file_path, 'r') as zf:

            # Extract all contents to the specified directory
            zf.extract(file_to_extract, destination_dir)
            if os.path.exists(os.path.join(destination_dir, file_to_extract)):
                if db: logger.debug(f"\tfile extracted successfully.")
                return True
            else:
                if db: logger.debug(f"\tfile could not be extracted.")
                return False
            #logger.debug(f"\tExtracted file from jar.")
            #logger.debug(f"\tFile extracted: '{file_to_extract}'")

    else:
        if db: logger.warning(f"file not found in .jar.")
        return False


def read_file_in_jar(jar_name, file_to_read) -> list[str] | None:
    logger.debug("")
    logger.debug("read_file_in_jar(jar_name, file_to_read)")
    logger.debug(f"\t-jar_name: {jar_name}")
    logger.debug(f"\t-file_to_read: {file_to_read}")

    # Open the JAR file using a context manager
    try:
        with zipfile.ZipFile(jar_name, 'r') as zf:
            # Read the file's content as a string
            # It's important to decode the bytes to a string (e.g., using 'utf-8')
            try:
                content_bytes = zf.read(file_to_read)
                content_text = content_bytes.decode('utf-8')
            except KeyError:
                print(f"Error: File '{file_to_read}' not found in the JAR archive.")
                return None
            except Exception as e:
                print(f"Error reading file content: {e}")
                return None

            # Iterate through the lines and search for texts
            lines = content_text.splitlines()  # Split the content into lines
            # for line_num, line in enumerate(lines, 1):
            #     for text in search_texts:
            #         if text in line:
            #             found_results[text].append((line_num, line.strip()))

    except zipfile.BadZipFile:
        print(f"Error: '{jar_name}' is not a valid JAR/ZIP file.")
        return None
    except FileNotFoundError:
        print(f"Error: '{jar_name}' not found.")
        return None

    return lines


def check_in_jar(jar_file_path, file_to_extract, db: bool = True) -> bool:
    if db:
        logger.debug("")
        logger.debug("check_in_jar(jar_file_path, file_to_extract)")
        logger.debug(f"\tjar_file_path:   '{jar_file_path}'")
        logger.debug(f"\tfile_to_extract: '{file_to_extract}'")
    try:
        #logger.debug(f"\tChecking if file in JAR...")
        with zipfile.ZipFile(jar_file_path, 'r') as zip_ref:
            # namelist() returns a list of all files/directories in the archive
            file_list = zip_ref.namelist()
            return file_to_extract in file_list
    except zipfile.BadZipFile:
        logger.error(f"\t'{jar_file_path}' is not a valid JAR file.")
        return False
    except FileNotFoundError:
        if not file_to_extract.endswith(".png.mcmeta"):
            if db: logger.warning(f"\tFile not found in JAR. file_to_extract: '{file_to_extract}'")
        elif "minecraft" not in file_to_extract:
            if db: logger.warning(f"\t\t\tWarning: File not found in JAR. file_to_extract: '{file_to_extract}")
            if db: logger.debug(f"\t\t\tFalling back to minecraft texture...")
        else:
            if db: logger.warning(f"\tFallback texture file not found in JAR. file_to_extract: '{file_to_extract}")
        return False


def get_mod_jar(modId: str, db: bool = True) -> str:
    if db:
        logger.debug("")
        logger.debug(f"get_mod_jar(modId)")
        logger.debug(f"\tmodId: {modId}")
    # if looking for something like 'hexarei' in settings, if its not in the modlist, it wont be found in settings
    # means there wasnt a blockstates directory for it so wasnt added
    # can still have models that we are looking for
    # have to manually search through all mod packs for modId.
    # TODO probably should write a file with modIds listed to refer to.
    # TODO needs to set once per script
    #sJson = get_settings_mod_paths()
    sJson = g_settings_mod_paths
    if 'mod_paths' in sJson:
        #TODO can i just get by key?
        if modId in sJson['mod_paths']:
            jar_path: str = sJson['mod_paths'][modId]
            if db: logger.debug(f"\tjar_path of modId: {jar_path}")
            #logger.debug(f"test check if get value by key: ")
            return jar_path

        elif "nb|" + modId in sJson['mod_paths']:
            jar_path: str = sJson['mod_paths']["nb|" + modId]
            if db: logger.debug(f"\tjar_path of nb|modId: {jar_path}")
            #logger.debug(f"test check if get value by key: ")
            return jar_path

        else:
            if db: logger.warning(f"\tmodId: {modId} NOT in mod_paths")
        # for key, val in sJson['mod_paths'].items():
        #     key: str = key.replace("nb|", "")
        #     #logger.debug(f"[get_mod_jar()]comparing modId: {modId} to key: {key}")
        #     if modId == key:
        #         logger.debug(f"[get_mod_jar()] returning val: {val}")
        #         return val

    # logger.debug(f"[get_mod_jar()] returning empty string.")
    # logger.error(f"Cannot find mod '{modId}' in imported mods.")
    # logger.info(f"Cannot find mod '{modId}' in imported mods.")
    #TODO cannot use these here since they will display when running with parse_model, use somewhere else
    #exit(1)
    return ""


def get_settings_mod_paths() -> Any | None:
    logger.debug("")
    logger.debug("get_settings_mod_paths()")
    settings_json: str = os.getcwd() + "\\assets\\settings.json"
    logger.debug(f"settingsJson: {settings_json}")
    if os.path.exists(settings_json):
        with open(settings_json, "r") as f:
            logger.debug("\t\tSettings file exists, returning json.load(f)")
            sJson = json.load(f)
            return sJson
            # if 'mod_paths' in sJson:
            #     for key, val in sJson['mod_paths'].items():
            #         logger.debug(f"key:{key}, value:{val}")
    return None


def getDependencies() -> list[str] | None:
    logger.debug("")
    logger.debug("getDependencies()")

    # TODO multiple mod types need addressed fabric, etc
    file: str = "META-INF/neoforge.mods.toml"
    modId: list[str] = []
    lines: list[str] = read_file_in_jar(mod_jar_path, file)

    for line in lines:
        if "modId=" in line:
            modId.append(line.replace("modId=", "")
                         .replace("\"", "")
                         .replace("'", "")
                         .replace("#mandatory", "")
                         .strip())

    logger.debug(f"modId: {modId}")
    return modId


def check_in_list(item: str, item_list: list[str]) -> bool:
    for i in item_list:
        if item == i: return True
    return False


def check_dependencies_for_texture(modname: str, val: str, texture_path: str) -> str:
    # comes in without 'block' in name
    # modname is regisrty name eg: modname:block_texture eg: "texture": "deeperdarker:block/echo_planks"
    logger.debug("")
    logger.debug(f"check_dependencies_for_texture(modname, val, texture_path)")
    logger.debug(f"\tmodname:      '{modname}'")
    logger.debug(f"\tval:          '{val}'")
    logger.debug(f"\ttexture_path: '{texture_path}'")

    # check if modname in dependencies first then loop through it all
    # get all dependencies of eg: bibliocraft if looking for deeperdarker:block/echo_planks
    dependency: str
    my_modId: str = ""
    dependencies: list[str] = getDependencies()
    # does modname ever change? why would this work?
    # if modname is deeperdarker, it may not be found in bibliocraft, so check in dependencies
    # if deeperdarker in bibliocraft_dependencies_list and deeperdarker not equal to first dependency(bibliocraft)
    # if modname(deeperdarker) does not equal bibliocraft, the first modId is our modId

    # TODO this may not work, modname in dependencies, yes but modname not equals the first one. only if in a loop?
    # modname != dependencies[0]: not equal to self
    if modname in dependencies and modname != dependencies[0]:
        # put 'block' back in texture_path
        logger.debug(f"\t-----------------------------------------------------")
        logger.debug(f"\tdependencies[0]: {dependencies[0]}")
        logger.debug(f"\tmodname: {modname}")

        texture_path = os.path.join(mod_assets_path, modname, "textures", "block")
        logger.debug(f"\ttexture_path: {texture_path}")
        #logger.debug(f"put 'block' back in texture_path. texture_path: {texture_path}")
        #TODO get_mod_jar(modname) will return jar file or empty strings if nothing found
        # should this fail if get mod jar is empty?

        # open texture file with new_jar_path set
        new_mod_jar: str = get_mod_jar(modname)
        if new_mod_jar != "":
            logger.error(f"\tjar for modname not found in modsList")
        else:
            if open_texture_file(modname, texture_path, val, new_mod_jar, False):
                open_texture_result = True
                logger.debug(f"\ttexture found!")
                return texture_path
            else:
                # take 'block' back out of texture_path
                texture_path = os.path.join(mod_assets_path, modname, "textures")
                logger.debug(f"\ttake 'block' back out of texture_path. texture_path: {texture_path}")
                if open_texture_file(modname, texture_path, val, new_mod_jar, False):
                    open_texture_result = True
                    return texture_path
                else:
                    logger.warning(
                        f"\ttexture file not found trying with both block and without. Checking the rest of the dependencies...")
                    pass
    logger.warning(f"\t-----------------------------------------------------")
    logger.warning(f"\tdidnt hit????")
    # TODO is this neccessary?
    # check every other modId's jar for that texture_path??
    logger.debug(f"\t trying for dependency in dependencies:")
    for dependency in dependencies:
        if my_modId == "":
            my_modId = dependency
            continue

        # avoid checking again since we already did up top
        #if dependency == modname: continue
        if dependency == 'minecraft': continue

        mod_jar: str = get_mod_jar(dependency, False)

        logger.debug(f"\t\tdependency: {dependency}")
        logger.debug(f"\t\tmod_jar:    {mod_jar}")
        texture_path = os.path.join(mod_assets_path, dependency, "textures", "block")

        if open_texture_file(modname, texture_path, val, mod_jar, False):
            logger.debug(f"\t\ttexture found!")
            logger.debug(f"\t\treturning texture_path. texture_path: {texture_path}")
            return texture_path
        else:
            texture_path = os.path.join(mod_assets_path, dependency, "textures")
            if open_texture_file(modname, texture_path, val, mod_jar, False):
                logger.debug(f"\t\ttexture found!")
                logger.debug(f"\t\treturning texture_path. texture_path: {texture_path}")
                return texture_path

        logger.debug(f"\t\tno texture found for {dependency}")
        logger.debug("")
    logger.error(f"\tError: texture_path not found in any of the dependencies jars.")
    return ""


def find_anim_in_dir(anim_dir: str, ent_anims: dict, target_anim='') -> Any:
    for dirpath, _, file_names in os.walk(anim_dir):
        for file_name in file_names:
            full_file_path = os.path.join(dirpath, file_name)
            if full_file_path.endswith('.json'):
                try:
                    with open(full_file_path, 'r') as f:
                        if target_anim:
                            if target_anim in f.read():
                                return full_file_path
                        else:
                            for anim in ent_anims.values():
                                if anim in f.read():
                                    return full_file_path
                except IOError as e:
                    logger.error(f"Error reading file {full_file_path}: {e}")
    return ''

# get geo name and search for geo model and get json java model
# def find_geo_in_dir(anim_dir: str, ent_anims: dict) -> Any:
#     for dirpath, _, file_names in os.walk(anim_dir):
#         for file_name in file_names:
#             full_file_path = os.path.join(dirpath, file_name)
#             if full_file_path.endswith('.json'):
#                 try:
#                     with open(full_file_path, 'r') as f:
#                         for anim in ent_anims.values():
#                             if anim in f.read():
#                                 return full_file_path
#                 except IOError as e:
#                     logger.error(f"Error reading file {full_file_path}: {e}")
#     return None

def get_closest_value_in_list(t: list, key: float) -> float:
    return min(t, key=lambda x: abs(x - key))


def get_time_in_frames(t_frames: list, time_xyz: list):
    """return time in time_frames where key is closest"""
    xyz: list = time_xyz[1]
    key_time: float = float(time_xyz[0])
    cv: float = get_closest_value_in_list(t_frames, key_time)
    return {'index': t_frames.index(cv), 'time': cv}


def parse_animation(bones: dict) -> bool:
    logger.debug("")
    logger.debug("<----------------parse_animation------------------->")
    logger.debug("")

    def log_output():
        logger.debug("")
        logger.debug("<--------------end parse_animation----------------->")
        logger.debug("")

    logger.debug(json.dumps(bones, indent=2))
    #entity_path: str = r"C:\Users\statiic\Desktop\storage_plus_resourcepack\entity\giggle\se\drawer\acacia_wide_drawer.entity.json"
    #entity_path: str = r"C:\Users\statiic\Desktop\storage_plus_resourcepack\entity\giggle\se\drawer\acacia_chest.entity.json"
    entity_path = geo_path
    #TODO
    # sort out these paths. this one still active while creating geomodel class will make bone from the geo_path intput.
    # needs to be cleaned up and working again.

    entity_json: Any = None
    # get by entity, will open and set data
    # might be better to do in java and convert to a json to handle everything
    if os.path.exists(entity_path):
        with open(entity_path) as ff:
            logger.debug("entity_path exists, entity_json = json.load(f)")
            entity_json = json.load(ff)

    #logger.debug(f"materials: {entity_json['minecraft:client_entity']['description']['materials']['default']}")
    description = entity_json['minecraft:client_entity']['description']
    ent_id: str = description['identifier']
    global ent_tex
    ent_tex = description['textures']['default']
    ent_geo: str = description['geometry']['default']
    # find json in dir where this is contained, get name of file, search for converted.json
    # or do we already have the name from the path we imported with?
    # its the model/entity that contains the build elements
    ent_anims: dict = description['animations']
    ent_scripts_anim: list = description['scripts']['animate']

    anim_dir: str = os.path.join(entity_path.split('entity')[0], "animations")
    #anim_path: str = find_anim_in_dir(anim_dir, ent_anims)
    anim_qc = LineBuilder()

    # format anim smd path
    mdl_model: str
    if '/' in json_model:
        #TODO will this only split if one '/' in name? shouldnt it be the last index of, or split by dirname?
        mdl_model = json_model.split("/")[1]
    else: mdl_model = json_model
    anims_path: str = os.path.join('anims', mdl_model)
    max_anim_time: float = 10.0

    frames_time: list = [0.04, 0.08, 0.13, 0.17, 0.21, 0.25, 0.29, 0.33, 0.38, 0.42, 0.46, 0.5,
                         0.54, 0.58, 0.62, 0.67, 0.71, 0.75, 0.79, 0.83, 0.87, 0.92, 0.96, 1.0]

    if not ent_anims: return False

    for name, anim in ent_anims.items():
        anim_json: Any = None
        anim_path: str = find_anim_in_dir(anim_dir, ent_anims, anim)
        logger.debug(f"anim_path: {anim_path}")

        if os.path.exists(anim_path):
            with open(anim_path) as ff:
                logger.debug("anim_path exists, anim_json = json.load(f)")
                logger.debug(f"\t[JSON_Path]:{anim_path}")
                anim_json = json.load(ff)
        else:
            continue
        if anim_json is None:
            continue
            #return False
        if 'animations' in anim_json:
            #for name, anim in ent_anims.items():
            if anim in anim_json['animations']:
                anim_long_name: str = anim
                anim_short_name: str = name
                logger.debug(f"anim: {anim_short_name} - "
                             f"{anim_json['animations'][anim]}")
                #anim_loop: str = anim_json['animations'][anim]['loop']
                if 'animation_length' not in anim_json['animations'][anim]:
                    logger.warning("Animation does not have a length. Skipping...")
                    continue

                anim_length: float = anim_json['animations'][anim]['animation_length']
                if anim_length > max_anim_time:
                    logger.error(f"{anim_short_name} - anim length: {anim_length} is higher then max anim length of {max_anim_time}. Anim trimmed to {max_anim_time}")
                    anim_length = max_anim_time

                anim_bones: Any = anim_json['animations'][anim]['bones']

                anim_smd = LineBuilder()
                anim_smd("version 1")
                anim_smd("nodes")
                anim_smd(make_smd_bones(bones))
                anim_smd("end")
                anim_smd("skeleton")

                #TODO generate only frames needed
                time_frames: list = [0.0]
                num_add: int = 1

                for ft in frames_time:
                    time_frames.append(ft)

                anim_t: float = get_closest_value_in_list(frames_time, anim_length)
                logger.debug(f"anim_t: {anim_t}")
                run_loop = True
                while run_loop:
                    # if length goes over 1 second
                    if anim_t == 1.0*num_add:
                        num_add += 1
                        # clear and start over
                        time_frames.clear()
                        # add 1 second and try again
                        for i in range(num_add):
                            for ft in frames_time:
                                time_frames.append(ft+i)
                        anim_t = get_closest_value_in_list(time_frames, anim_length)
                    else:
                        run_loop = False
                logger.debug(f"anim_t: {anim_t}")
                #logger.debug(f"time_frames: {time_frames}")

                bone_rotations: list[list] = []
                bone_positions: list[list] = []

                anim_bones_dict: dict = anim_bones
                # combined bone positions and bone rotations and clear
                yet_another_bones_list: list[list] = []
                anim_time: float = get_closest_value_in_list(time_frames, anim_length)
                #TODO
                # all bones need defined even if all zero
                # need all undefined bones to have default position and rotation
                # from Bone constructer
                # need a list of all bones, if not in anims, put default position/rotation
                # AND/OR...
                # bone need its rotation set when setting up model in elements
                # a rotation of -85(bbq model) will show as 0 angle and animations are backwards because of it

                for k, b in anim_bones_dict.items():
                    bone = b
                    logger.debug(f"bone: {k}")
                    logger.debug(f"k: {k}, b: {b}")
                    bone_name = k
                    bone_id = get_bone_id_by_name(bones, bone_name)
                    logger.debug(f"g_test_bones.len: {len(bones)}")
                    logger.debug(f"bone_id: {bone_id}")
                    bone_pos: Any
                    bone_origins:list = get_bone_origins(bones)
                    tpos: list[float] = bone_origins[bone_id]

                    if 'rotation' not in bone:
                        for i in time_frames:
                            if i <= anim_time:
                                bone_rotations.append([])
                        for i in time_frames:
                            if i <= anim_time:
                                ind: int = indexOf(time_frames, i)
                                rot: list = [0.0, 0.0, 0.0]
                                new_rot: list = [0, 0, 0]
                                new_rot[0], new_rot[1], new_rot[2] = (
                                    math.radians(rot[0]), math.radians(rot[2]), math.radians(-rot[1]))
                                bone_rotations[ind] = ([i, bone_id, new_rot, 'rot'])
                    else:
                        bone_rotation: dict = bone['rotation']
                        # populate list with empty list, then replace with actual values if exist at time
                        for i in time_frames:
                            if i <= anim_time+1:
                                bone_rotations.append([])
                        try:
                            # multiple lists
                            # eg: "0.0": [0, 0, 0],
                            #     "0.0833": [0, 0, -0.67],
                            for i in time_frames:
                                for time, rot in bone_rotation.items():
                                    time_in_frames: dict = get_time_in_frames(time_frames, [time, rot])
                                    new_time: float = time_in_frames.get('time')
                                    if new_time == i:
                                        ind: int = indexOf(time_frames, i)
                                        new_rot: list = [0, 0, 0]

                                        if isinstance(rot, dict):
                                            if 'pre' in rot:
                                                rot = rot['pre']
                                            elif 'post' in rot:
                                                rot = rot['post']
                                            logger.warning(f"pre/post in rotation. '{i}', '{k}', '{b}', '{time}', '{rot}'")

                                        # check if any x, y or z use molang expressions
                                        for xyz in range(3):
                                            if isinstance(rot[xyz], str):
                                                logger.error(f"contains math expression and is not yet supported. '{rot[xyz]}'")
                                            # else:
                                            #     logger.error(f"rot[xyz]: {rot[xyz]}")

                                        # flip y and z for source engine orientation and covert to radians
                                        new_rot[0], new_rot[1], new_rot[2] = (
                                            math.radians(rot[0]), math.radians(rot[2]), math.radians(-rot[1]))
                                        logger.debug(f"ind: {ind}")
                                        logger.debug(f"len of bone_rotatons: {len(bone_rotations)}")
                                        logger.debug(f"new_time: {new_time}, bone_id: {bone_id}, new_rot: {new_rot}")
                                        bone_rotations[ind] = [new_time, bone_id, new_rot, 'rot']
                        # except KeyError:
                        #     logger.error(f"KeyError triggered. '{k}', '{b}' Skipping keyframe for now...")
                        #     pass
                        except AttributeError:
                            # only one value for all
                            # 'rotation':[0, 0, 0]
                            time, rot = 0.0, [0, 0, 0]
                            for i in time_frames:
                                if i <= anim_time:
                                    ind: int = indexOf(time_frames, i)
                                    new_rot: list = [0, 0, 0]
                                    # flip y and z for source engine orientation and covert to radians
                                    new_rot[0], new_rot[1], new_rot[2] = (
                                        math.radians(rot[0]), math.radians(rot[2]), math.radians(-rot[1]))

                                    for xyz in range(3):
                                        if isinstance(bone_rotation[xyz], str):
                                            logger.error(f"contains math expression and is not yet supported. '{bone_rotation[xyz]}'")
                                        # else:
                                        #     logger.error(f"rot[xyz]: {bone_rotation[xyz]}")

                                    bone_rotations[ind] = ([i, bone_id, new_rot, 'rot'])

                    if 'position' not in bone:
                        # populate list with empty list, then replace with actual values if exist at time
                        for i in time_frames:
                            if i <= anim_time:
                                bone_positions.append([])
                        #time, pos = 0.0, bone_position
                        for i in time_frames:
                            if i <= anim_time:
                                ind: int = indexOf(time_frames, i)
                                pos = [0.0, 0.0, 0.0]
                                pos[0] += tpos[0]
                                pos[1] += tpos[1]
                                pos[2] += tpos[2]
                                bone_positions[ind] = ([i, bone_id, pos, 'pos'])
                    else:
                        bone_position: dict = bone['position']
                        # populate list with empty list, then replace with actual values if exist at time
                        for i in time_frames:
                            if i <= anim_time+1:
                                bone_positions.append([])
                        try:
                            # multiple lists
                            # eg: "0.0833": [0, 0, 0],
                            # 	  "0.125": [2.5, 0, 0],
                            for i in time_frames:
                            #for i in bone_positions:
                                for time, pos in bone_position.items():
                                    # if isinstance(pos, dict):
                                    #     logger.error(f"KeyError triggered in position. '{i}', '{k}', '{b}', '{time}', '{pos}' Skipping keyframe for now...")
                                    #     continue
                                    time_in_frames: dict = get_time_in_frames(time_frames, [time, pos])
                                    new_time: float = time_in_frames.get('time')
                                    #if new_time == time_frames[indexOf(bone_positions, i)]:
                                    if new_time == i:
                                        #ind: int = indexOf(bone_positions, i)
                                        ind: int = indexOf(time_frames, i)
                                        if isinstance(pos, dict):
                                            if 'pre' in pos:
                                                pos = pos['pre']
                                            elif 'post' in pos:
                                                pos = pos['post']
                                            logger.warning(f"pre/post in position. '{i}', '{k}', '{b}', '{time}', '{pos}'")

                                        # check if any x, y or z use molang expressions
                                        for xyz in range(3):
                                            if isinstance(pos[xyz], str):
                                                logger.error(f"contains math expression and is not yet supported. '{bone_position[xyz]}'")
                                            # else:
                                            #     logger.error(f"pos[xyz]: {bone_position[xyz]}")

                                        # flip z and y for source engine orientation
                                        pos[0], pos[1], pos[2] = pos[0], pos[2], pos[1]
                                        #logger.debug(f"before - pos: {pos}, tpos: {tpos}")
                                        pos[0] += tpos[0]
                                        pos[1] += tpos[1]
                                        pos[2] += tpos[2]
                                        #logger.debug(f"after - pos: {pos}, tpos: {tpos}")
                                        bone_positions[ind] = [new_time, bone_id, pos, 'pos']
                                        logger.debug(f"bone_positions[ind]: {bone_positions[ind]}")

                            #TODO
                            # do this on interp, last frames if null, stretch from last key to lst frame
                            #for bp in bone_positions:
                            #     logger.debug(f"bp: {bp}")
                            #     if not bp:
                            #         gg = indexOf(bone_positions, bp)
                            #         bone_positions[gg] = bone_positions[gg-1]
                                # should i replace 'for i in time_frames:' with bone_positions?
                                # block is finishing with 2 frames left as []
                                # anim time is .5, 2 frames more then last key time(0.4167)
                                # needs to finish regardless of what the last key is
                        # except KeyError:
                        #     logger.error(f"KeyError triggered. '{k}', '{b}' Skipping keyframe for now...")
                        #     pass
                        except AttributeError:
                            # only one value for all
                            # 'position':[0, 0, 0]
                            time, pos = 0.0, tpos
                            # insert value acrossed all keys
                            for i in time_frames:
                                if i <= anim_time:
                                    ind: int = indexOf(time_frames, i)
                                    #pos[0], pos[1], pos[2] = pos[0], pos[2], pos[1]
                                    # check if any x, y or z use molang expressions
                                    for xyz in range(3):
                                        if isinstance(bone_position[xyz], str):
                                            logger.error(f"contains math expression and is not yet supported. '{pos[xyz]}'")
                                        # else:
                                        #     logger.error(f"pos[xyz]: {pos[xyz]}")

                                    bone_positions[ind] = ([i, bone_id, pos, 'pos'])

                    # combined position and rotations
                    bone_frames: list[list] = []
                    for tf in time_frames:
                        if tf <= anim_time:
                            entry_line: list = [tf, bone_id, [], []]
                            rot_entry_line: list = []
                            pos_entry_line: list = []

                            for rot_entry in bone_rotations:
                                # if time == time_frames time at tf
                                if len(rot_entry) > 0:
                                    if rot_entry[0] == tf:
                                        # add to array
                                        rot_entry_line = rot_entry

                            for pos_entry in bone_positions:
                                # if time == time_frames time at tf
                                if len(pos_entry) > 0:
                                    if pos_entry[0] == tf:
                                        # add to array
                                        pos_entry_line = pos_entry

                            if len(rot_entry_line) > 0:
                                entry_line[0] = rot_entry_line[0]
                                entry_line[1] = rot_entry_line[1]
                                if len(pos_entry_line) == 0:
                                    entry_line[2] = []
                                entry_line[3] = rot_entry_line[2]

                            if len(pos_entry_line) > 0:
                                entry_line[0] = pos_entry_line[0]
                                entry_line[1] = pos_entry_line[1]
                                entry_line[2] = pos_entry_line[2]
                                if rot_entry_line == 0:
                                    entry_line[3] = []
                            bone_frames.append(entry_line)
                        else:
                            pass
                            #logger.debug(f"tf > anim_time")

                    # interpolate lines before entry into list
                    last_key_pos: int = -1
                    last_key_rot: int = -1
                    null_frames_pos: list = []
                    null_frames_rot: list = []
                    for bfs in bone_frames:
                        #logger.debug(f"bfs: {bfs}")
                        e: int = indexOf(bone_frames, bfs)
                        bf_pos: list = bfs[2]
                        bf_rot: list = bfs[3]

                        if len(bf_pos) > 0:
                            if last_key_pos != -1:
                                # get frames between
                                ncp = len(null_frames_pos)
                                for n in null_frames_pos:
                                    # if last frame was last key in anim
                                    # need to convert it out

                                    for v in range(3):
                                        current_frame = bone_frames[e][2]
                                        last_frame = bone_frames[last_key_pos][2]
                                        add_num: float = (current_frame[v] - last_frame[v]) / (ncp + 1)
                                        # bone_frames[null_frame][pos][z] =
                                        f_count = indexOf(null_frames_pos, n) + 1
                                        #logger.debug(f"v: {v}")
                                        #logger.debug(f"current_frame: {current_frame[v]}")
                                        #logger.debug(f"last_frame: {last_frame[v]}")
                                        #logger.debug(f"add_num: {add_num}")
                                        #logger.debug(f"f_count: {f_count}")
                                        if not bone_frames[n][2]:
                                            bone_frames[n][2] = [None, None, None]

                                        # (current_frame - last_key_pos) / num frames to lerp for + 1
                                        bone_frames[n][2][v] = round(last_frame[v] + (add_num * f_count), 5)
                                        #logger.debug(f"bone_frames[n][2][v]: {bone_frames[n][2][v]}")
                                null_frames_pos.clear()
                            last_key_pos = e
                        else:
                            #if last_key_pos > anim_time:
                            logger.debug(f"Position - last_key_pos: {last_key_pos},anim_time: {anim_time},"
                                         f" e: {e}, len(bone_frames): {len(bone_frames)}")
                            #if e+1 == len(bone_frames):
                                # at end with null frames
                                # take last key pos and apply it to all null to stratch it out to end
                            null_frames_pos.append(e)
                            if e+1 == len(bone_frames):
                                logger.debug(f"e+1 == len(bone_frames) in pos")
                                for n in null_frames_pos:
                                    if not bone_frames[n][2]:
                                        bone_frames[n][2] = [None, None, None]
                                    bone_frames[n][2] = bone_frames[last_key_pos][2]
                                null_frames_pos.clear()

                        if len(bf_rot) > 0:
                            if last_key_rot != -1:
                                # get frames between
                                ncp = len(null_frames_rot)
                                for n in null_frames_rot:
                                    for v in range(3):
                                        current_frame = bone_frames[e][3]
                                        last_frame = bone_frames[last_key_rot][3]
                                        add_num: float = (current_frame[v] - last_frame[v]) / (ncp + 1)
                                        # bone_frames[null_frame][pos][z] =
                                        f_count = indexOf(null_frames_rot, n) + 1
                                        #logger.debug(f"v: {v}")
                                        #logger.debug(f"current_frame: {current_frame[v]}")
                                        #logger.debug(f"last_frame: {last_frame[v]}")
                                        #logger.debug(f"add_num: {add_num}")
                                        #logger.debug(f"f_count: {f_count}")
                                        if not bone_frames[n][3]:
                                            bone_frames[n][3] = [None, None, None]

                                        # (current_frame - last_key_pos) / num frames to lerp for + 1
                                        bone_frames[n][3][v] = round(last_frame[v] + (add_num * f_count), 5)
                                        #logger.debug(f"bone_frames[n][3][v]: {bone_frames[n][3][v]}")
                                null_frames_rot.clear()
                            last_key_rot = e
                        else:
                            logger.debug(f"Rotation - last_key_pos: {last_key_pos},anim_time: {anim_time},"
                                         f" e: {e}, len(bone_frames): {len(bone_frames)}")
                            null_frames_rot.append(e)
                            if e+1 == len(bone_frames):
                                logger.debug(f"e+1 == len(bone_frames) in rot")
                                for n in null_frames_rot:
                                    if not bone_frames[n][3]:
                                        bone_frames[n][3] = [None, None, None]
                                    bone_frames[n][3] = bone_frames[last_key_rot][3]
                                null_frames_rot.clear()

                    yet_another_bones_list.append(bone_frames)
                    # clear for each new bone
                    bone_rotations.clear()
                    bone_positions.clear()

                before_logs = """
            yet_another_bones_list:
[DEBUG]:			bone_frames:
[DEBUG]:				[0.0, 2, [0, 0, 0], [0, 0, 0]]
[DEBUG]:				[0.04, 2, [], []]
[DEBUG]:				[0.08, 2, [0, 0, -0.67], [0, 0, 0]]
[DEBUG]:				[0.13, 2, [], [2.5, 0, 0]]
[DEBUG]:				[0.17, 2, [], []]
[DEBUG]:				[0.21, 2, [], []]
[DEBUG]:				[0.25, 2, [], []]
[DEBUG]:				[0.29, 2, [], []]
[DEBUG]:				[0.33, 2, [], []]
[DEBUG]:				[0.38, 2, [0, 0, -11], [0, 0, 0]]
[DEBUG]:				[0.42, 2, [0, 0, -11.325], [0, 0, 0]]
                """

                logger.debug(before_logs)

                logger.debug("")
                logger.debug(f"yet_another_bones_list:")
                for yabl in yet_another_bones_list:
                    logger.debug(f"\tbone_frames:")
                    for bf in yabl:
                        logger.debug(f"\t\t{bf}")

                logger.debug("")
                ttimes_list: list[list] = []
                for yabl in yet_another_bones_list:
                    for i in time_frames:
                        if i <= anim_time:
                            time_i = indexOf(time_frames, i)
                            #logger.debug(f"time_i: {time_i}")
                            if len(ttimes_list) < time_i + 1:
                                ttimes_list.append([])
                            #logger.debug(f"time {time_i}")
                            for bf in yabl:
                                if bf[0] == i:
                                    # bf add bone refs
                                    if bf[1] == 1:
                                        pass  #TODO
                                        #bf[2] = [] # pos
                                    ttimes_list[time_i].append(bf)

                bone_origins:list = get_bone_origins(bones)
                for ttl in ttimes_list:
                    ttl.sort(key=lambda x: x[1])
                    ttt: int = indexOf(ttimes_list, ttl)
                    logger.debug(f"time {ttt}")
                    anim_smd(f"time {ttt}")
                    do_once: bool = False
                    t2_set: bool = False
                    for t in ttl:
                        logger.debug(f"t: {t}")
                        if not do_once:
                            #logger.debug(f"do once t: {t}")
                            do_once = True
                            #anim_smd(f"0  0.0 0.0 0.0  0.0 0.0 0.0")

                            #perfect fit now
                            # origin or root must be the same, if not moving?
                            # must all bone origins be added? or remain the same?
                            tpos: Vector = bone_origins[0]
                            #logger.debug(f"0 - tpos: {tpos}")
                            anim_smd(f"0  {tpos[0]} {tpos[1]} {tpos[2]}  0.0 0.0 0.0")
                            #anim_smd(f"0  7.0 -7.0 1.0  0.0 0.0 0.0")  # try adding idle ref pos rots to anims


                        # if t[1] == 1:
                        #     tpos: list = bone_origins[t[1]]
                        #     #logger.debug(f"1 - tpos: {tpos}")
                        #     # t[2][0] = tpos[0]
                        #     # t[2][1] = tpos[1]
                        #     # t[2][2] = tpos[2]
                        #     # t[2][0] = -7.5
                        #     # t[2][2] = -1.0
                        #     # t[2][1] = -0.05
                        # if t[1] == 2:
                        #     tpos: list = bone_origins[t[1]]
                        #     #logger.debug(f"2 - tpos: {tpos}")
                        #     # t[2][0] -= tpos[0]
                        #     # t[2][1] -= tpos[1]
                        #     # t[2][2] -= tpos[2]
                        #     # t[2][0] += -7.0
                        #     # t[2][2] += -1.0
                        #     # t[2][1] += 0.0
                        # if t[1] == 3:
                        #     tpos: list = bone_origins[t[1]]
                        #     #logger.debug(f"3 - tpos: {tpos}")
                        #     # t[2][0] -= tpos[0]
                        #     # t[2][1] -= tpos[1]
                        #     # t[2][2] -= tpos[2]
                        #     # t[2][0] = -7.0
                        #     # t[2][2] = -1.0
                        #     # t[2][1] = -0.025
                        # if t[1] == 4:
                        #     tpos: list = bone_origins[t[1]]
                        #     #logger.debug(f"4 - tpos: {tpos}")
                        #     # t[2][0] -= tpos[0]
                        #     # t[2][1] -= tpos[1]
                        #     # t[2][2] -= tpos[2]
                        #     # t[2][0] = -7.0
                        #     # t[2][2] = 3.5
                        #     # t[2][1] = 1.5
                        # if t[1] == 1:
                        #     t[2][0] = -7.5 - 7.0
                        #     t[2][2] = -1.0 - 1.0
                        #     t[2][1] = -0.05 - -7.0
                        # if t[1] == 2:
                        #     t2_set = True
                        #     t[2][0] += (-7.0 - 7.0)
                        #     t[2][2] += (-1.0 - 1.0)
                        #     t[2][1] += (0.0 - -7.0)
                        # if t[1] == 3:
                        #     t[2][0] = -7.0 - 7.0
                        #     t[2][2] = -1.0 - 1.0
                        #     t[2][1] = -0.025 - -7.0
                        # if t[1] == 4:
                        #     t[2][0] = -7.0 - 7.0
                        #     t[2][2] = 3.5 - 1.0
                        #     t[2][1] = 1.5 - -7.0

                        anim_smd(f"{t[1]}  "
                                 f"{round(t[2][0], 5)} {round(t[2][1], 5)} {round(t[2][2], 5)}  "
                                 f"{round(t[3][0], 5)} {round(t[3][1], 5)} {round(t[3][2], 5)}")
                        #logger.debug(t)

                logger.debug("")

                # anim_smd("time 0")
                # anim_smd("0  7.0 -7.0 1.0  0 0 0.0")
                # anim_smd("1  -7.5 -0.04999999999999982 -1.0  0 0 0.0")
                # anim_smd("2  -7.0 0.0 -1.0  0 0 0.0")
                # anim_smd("3  -7.0 -0.025000000000000355 -1.0  0 0 0.0")
                # anim_smd("4  -7.0 1.5 3.5  0 0 0.0")
                #
                # anim_smd("time 1")
                # anim_smd("0  7.0 -7.0 1.0  0 0 0.0")
                # anim_smd("1  -7.5 -0.04999999999999982 -1.0  0 0 0.0")
                # anim_smd("2  -7.0 -11.0 -1.0  0 0 0.0")
                # anim_smd("3  -7.0 -0.025000000000000355 -1.0  0 0 0.0")
                # anim_smd("4  -7.0 1.5 3.5  0 0 0.0")

                anim_smd("end")
                anim_smd("triangles")
                anim_smd("end")

                logger.debug(f"\n{anim_smd}")

                #anims_path = os.path.join('anims', mdl_model, anim_short_name)
                # write anim smd
                fp = os.path.join(out_models_path, anims_path, anim_short_name + '.smd')
                os.makedirs(os.path.dirname(fp), exist_ok=True)
                os.makedirs(os.path.dirname(anims_path), exist_ok=True)
                with open(fp, 'w', encoding='utf-8') as f:
                    f.write(str(anim_smd))

                if os.path.exists(fp):
                    logger.debug(f"anim SMD successfully written to '{fp}'.")
                else:
                    logger.error(f"anim SMD could not be written.")

                # format anim_sequences.qci contents

                #anim_qc = LineBuilder()
                anim_qc(f'$sequence "{anim_short_name}" ' + '{')
                anim_qc(f'\t"{os.path.join(anims_path, anim_short_name)}"')
                anim_qc(f"\tfps 24")
                anim_qc("}")

            # write anim_sequences.qci. return true if written
            fp = os.path.join(out_models_path, 'anim_sequences' + '.qci')
            os.makedirs(os.path.dirname(fp), exist_ok=True)
            #os.makedirs(os.path.dirname(anims_path), exist_ok=True)

            with open(fp, 'w', encoding='utf-8') as f:
                logger.debug(f"\twith open(fp, 'w', encoding='utf-8') as f:")
                f.write(str(anim_qc))

            #testbones: GeoModel = GeoModel(entity_path)
            # logger.debug(f"parse_geometry():")
            # for tb in g_test_bones:
            #     #tb: dict = tb
            #     logger.debug(f"{tb}:{g_test_bones.get(tb)}")

            # logger.debug(f"testbones.get_bones_dict():")
            # for tb in testbones.get_bones_dict():
            #     #tb: dict = tb
            #     logger.debug(f"{tb}:{testbones.get_bones_dict().get(tb)}")
            #logger.debug(f"\n{g_test_bones}")
            #logger.debug(f"Test Geomodel bones: \n {testbones.get_bones_dict()}")

            if os.path.exists(fp):
                logger.debug(f"File successfully written to '{fp}'.")
                #log_output()
                #return True
            else:
                logger.error(f"anim_sequences.qci could not be written.")
                #log_output()
                #return False

    # before returning
    # make anim_sequences.qci
    # and write

    log_output()
    return True


def get_bone_id(test_bones: dict, elem_count: int):
    bone_id: int = 0
    for b in test_bones:
        bone: dict = b
        bb: dict = test_bones.get(bone)
        cubes = bb.get('cubes')
        for cube in cubes:
            if is_int(cube):
                if elem_count in cubes:
                    bone_id += indexOf(test_bones, bone)
                    return bone_id
    return bone_id


def get_bone_id_by_name(test_bones: dict, bone_name: str):
    bone_id: int = 0
    for b in test_bones:
        #bone: dict = b
        bone: dict = test_bones.get(b)
        if bone.get('name') == bone_name:
            return indexOf(test_bones, b)
    return None


def make_smd_bones(test_bones) -> str:
    smd_bones = LineBuilder()
    smd_bones_id: int = 0
    smd_bones_id_list: list = []
    bone_indices: list[str] = []
    for b in test_bones:
        def smd_end():
            """lineBuilder will end each line with '\n'. add only to last elements"""
            if indexOf(test_bones, b) == len(test_bones) - 1: return ''
            return '\n'

        bone: dict = test_bones.get(b)
        smd_bone_id: int = -1
        #if bone.get('parent') == bone.get('name'):
        #smd_bones_int += indexOf(test_bones, bone.get('parent'))
        parent: str = bone.get('parent')
        smd_bone_id += indexOf(test_bones, parent)
        bone_indices.append(parent)
        #smd_bone_id += indexOf(test_bones, bone.get('name')

        if bone.get('name') == bone.get('parent'):
            smd_bones(f'{smd_bones_id} "{bone.get('name')}" {-1}', smd_end())
        else:
            #smd_bones(f'{smd_bones_id} "{bone.get('name')}" {indexOf(bone_indices, parent)}', smd_end())
            smd_bones(f'{smd_bones_id} "{bone.get('name')}" {indexOf(smd_bones_id_list, bone.get('parent'))}', smd_end())

        smd_bones_id_list.append(bone.get('name'))
        smd_bones_id += 1
    return str(smd_bones)


def get_bone_origins(test_bones: dict):
    positions: list = []
    for b in test_bones:
        bone: dict = test_bones.get(b)
        smd_bone_name = bone.get('name')
        smd_bone_parent = bone.get('parent')
        #smd_bone_origin = Vector(*bone.get('origin'))
        smd_bone_origin = [bone.get('origin')[0], bone.get('origin')[1], bone.get('origin')[2]]

        #smd_offset = Vector(0, 0, 0)
        smd_offset = [0.0, 0.0, 0.0]
        if smd_bone_parent != smd_bone_name:
            #offset by parent origin
            bone_parent: dict = test_bones.get(smd_bone_parent)
            parent_origin = bone_parent.get('origin')
            #smd_offset = Vector(*parent_origin)
            smd_offset = [parent_origin[0], parent_origin[1], parent_origin[2]]

        #smd_bone_origin = smd_bone_origin - smd_offset
        smd_bone_origin[0] -= smd_offset[0]
        smd_bone_origin[1] -= smd_offset[1]
        smd_bone_origin[2] -= smd_offset[2]

        smd_bone_origin[0], smd_bone_origin[1], smd_bone_origin[2] = (
            smd_bone_origin[0], smd_bone_origin[2], smd_bone_origin[1])

        positions.append(smd_bone_origin)
    return positions


def handle_bone_placement(test_bones: dict) -> str:
    smd_anim_test = LineBuilder()
    smd_anim_test('time 0')
    for b in test_bones:
        def smd_end():
            """lineBuilder will end each line with '\n'. add only to last elements"""
            if indexOf(test_bones, b) == len(test_bones) - 1: return ''
            return '\n'

        bone: dict = test_bones.get(b)
        smd_bone_name = bone.get('name')
        smd_bone_parent = bone.get('parent')
        #smd_bone_origin = Vector(*bone.get('origin'))
        smd_origin = bone.get('origin')
        smd_bone_origin: list[float] = [smd_origin[0], smd_origin[1], smd_origin[2]]
        smd_rotation = Vector(0, 0, 0)
        if bone.get("rotation"):
            smd_rotation = Vector(bone.get("rotation")[0], bone.get("rotation")[1], bone.get("rotation")[2])

        #smd_offset = Vector(0, 0, 0)
        smd_offset = [0.0, 0.0, 0.0]
        if smd_bone_parent != smd_bone_name:
            # offset by parent origin
            bone_parent: dict = test_bones.get(smd_bone_parent)
            parent_origin = bone_parent.get('origin')
            #smd_offset = Vector(*parent_origin)
            smd_offset = [parent_origin[0], parent_origin[1], parent_origin[2]]

        smd_bone_origin[0] -= smd_offset[0]
        smd_bone_origin[1] -= smd_offset[1]
        smd_bone_origin[2] -= smd_offset[2]

        smd_bone_origin[0], smd_bone_origin[1], smd_bone_origin[2] = (
            smd_bone_origin[0], smd_bone_origin[2], smd_bone_origin[1])

        bone_index = indexOf(test_bones, b)
        smd_bone_origins: str = f"{smd_bone_origin[0]} {smd_bone_origin[1]} {smd_bone_origin[2]}"
        #TODO convert to radians?
        smd_bone_rotations: str = f"{math.radians(-smd_rotation.x)} {math.radians(smd_rotation.y)} {math.radians(smd_rotation.z)}"
        smd_anim_test(f"{bone_index}  {smd_bone_origins}  {smd_bone_rotations}", smd_end())
    return str(smd_anim_test)


def find_json_file(directory: str, filename: str):
    for dirpath, _, file_names in os.walk(directory):
        filename = filename.replace(".json", "")
        for file_name in file_names:
            file_name = file_name.replace(".json", "")
            full_file_path = os.path.join(dirpath, file_name + ".json")
            # works
            if 'Converted' in file_name and filename in file_name:
                logger.debug("json file found!")
                logger.debug(f"full_file_path: {full_file_path}")
                logger.debug(f"file_name: {file_name}, filename: {filename}")
                try:
                    logger.debug(f"true, trying to open...full_file_path: {full_file_path}")
                    with open(full_file_path, 'r') as f:
                        logger.debug(f"file can be opened.")
                        if f.readable():
                            return full_file_path
                except IOError as e:
                    logger.error(f"Error reading file {full_file_path}: {e}")
    return None


def parse_geo_model(models_path: str, model_file: str) -> List[str]:
    """
    Parse model json text file
    Parameters:
        models_path (str): model's path to model json file.
        model_file (str): model's json text file as inputted.
    """
    model_file = model_file.replace(" ", "_").replace(".", "_")
    logger.debug("")
    logger.debug(f'parse_model(models_path, model_file)')
    logger.debug(f"\tmodels_path: '{models_path}'")
    logger.debug(f"\tmodel_file: '{model_file}')")

    # Open file, if not found, try extracting from jar
    anim_sequences: str = '$staticprop'
    g_test_bones: dict = {}

    # will need to get the geo_path from the json file, entry we add
    # needs to get the entity json and check what the default[geometry] is
    # scan through files looking for that identifier in 'models\' and/or 'models\entity' directory
    # then get the name of the file. then get the converted version of it (in the same directory,
    # as instructed to where to place after exporting from blockbench)
    # on python start, will probably run a different parse_model first just to initiate the geo data
    # then once the json converted file is found, run parse_model() as normal
    # will have to remove everything that isn't needed or is run already in our first function

    #geo_path2: str = os.path.join(models_path, model_file)
    logger.debug(f"geo_path: {geo_path}")
    #exit(1)
    geo_model: GeoModel = GeoModel(geo_path)
    g_test_bones = geo_model.get_bones_dict()
    if len(g_test_bones) == 0:
        logger.error(f"Error: g_test_bones is empty.")
        exit(1)
    # return qc string section to add to qc
    if parse_animation(g_test_bones):
        anim_sequences = f'$include "anim_sequences.qci"'
        # split directory from filename
        geo_model_path: str = geo_model.geo_model_path
        directory, filename = os.path.split(geo_model_path)
        logger.debug(f"geo_model.geo_model_path: {geo_model.geo_model_path}")
        logger.debug(f"directory: {directory}, filename: {filename}")
        # find file which contains the filename. looking in example file: slimdrawer_3.geo.json, converted: slimdrawer.geo - Converted.
        # redefine as json model we found
        geo_j_model = find_json_file(directory, filename)
        if geo_j_model is not None:
            directory, filename = os.path.split(geo_j_model)
            models_path = directory
            model_file = filename.replace(".json","")
            #og_model_file = model_file
            #filename = filename.replace(".json","")
            logger.debug(f"models_path: {models_path}, model_file: {model_file}")
            # now need to go through and say if not is_geo true, to skip things like seardching for textures and whatnot
            # needs to get textures from entity file, textures.default.
            # make sure to make global and tell it to use that path instead
            #exit(1)
        else:
            logger.error(f"geo_j_model not found. directory: {directory}, filename: {filename}")
            exit(1)

    logger.debug(f"g_test_bones: {g_test_bones}")

    new_path: str = os.path.join(models_path, model_file + '.json')
    logger.debug(f"\tnew_path: {new_path}")
    jmodel = open_model_file(new_path, "")

    if jmodel is None:
        logger.error(f"Error: jmodel not found.")
        exit(1)
    else:
        logger.debug(f"\t[JSON_Path]:{new_path}")

    idx: int = model_file.find('/')
    model_name = model_file[idx + 1:]
    logger.debug(f"\tmodel_name = model_file[idx + 1:]: '{model_file[idx + 1:]}'")

    logger.debug(f'\tCreating new qc: {model_file}')
    qc = LineBuilder()

    textures = []
    ''' key for texture, eg: side '''
    undefined_textures = []
    ''' any texture(value in key:value) starting with # pointing to what tex to reference. 
        textures not defined in model_file? '''

    if 'model' in jmodel:
        #model: str = jmodel['model']
        if '.obj' in jmodel['model']:
            logger.error("\tmodel requires an .obj model and not currently supported.")
            exit(1)

    if 'textures' in jmodel:
        logger.debug("")
        logger.debug(f"if 'textures' in jmodel:")
        qc('// JSON "textures":')

        modname: str = ""
        for tex, val in jmodel['textures'].items():
            logger.debug(f"for tex, val in jmodel['textures'].items():")
            logger.debug(f"\ttex: {tex}")
            logger.debug(f"\tval: {val}")

            val = ent_tex
            sub_dirs: str = ""
            if "/" in val:
                logger.debug(f"if '/' in val:")
                sub_dirs = os.path.dirname(val) + "/"
                #sub_dirs = val.replace(new_val, "")
                new_val: str = val.split("/")[-1]  # alternator last occurrence
                #sub_dirs = val.replace(new_val, "") # alternator/alternator being replaced leaving '/'
                val = new_val
                logger.debug(f"\tval: '{val}'")
                logger.debug(f"\tsub_dirs: '{sub_dirs}'")

            # assets_path = "...\Minecraft MDL to Source MDL\assets_other\assets"
            logger.debug(f"\tval: '{val}', tex:'{tex}'")  # eg val: front, tex: birch_front
            #qc(f'$definevariable texture_{tex} "{val}"')

            #texture_path = os.path.join(assets_path, f'{modname}/textures/block/{sub_dirs}')
            #texture_path = os.path.join(assets_path, modname, "textures", "block", sub_dirs)

            val = os.path.join(sub_dirs, val)
            # if texture didnt have namespace: at beginning, this will be empty
            # fallback to mod_name of model,
            # could also be minecraft but check later in export textures
            if modname == "":
                modname = mod_name
            #texture_path = os.path.join(assets_path, modname, "textures", "block")
            # should path contain block ?
            # "0": "endermanoverhaul:entity/badlands/badlands_enderman",
            # is added here when it doesn't need it and fails to find texture
            # is first time in testing this as been a problem

            texture_path = os.path.join(
                mod_assets_path.replace("assets\\assets", "assets\\bedrock_assets"),
                modname)

            # if texture didnt have namespace: prefixed, then use default mod_name
            # may also be something else? maybe just minecraft?

            open_texture_result: bool = False
            if open_texture_file(modname, texture_path, val, ""):
                open_texture_result = True

            logger.debug(
                f"\topen_texture_file() -> returned: {open_texture_result}")  # try extracting if not exists

            # if not open_texture_result:
            #     # try again without block in path
            #     #TODO does this really need checked? does block really sometimes need added?
            #     logger.debug(f"\tRetrying open_texture_file() without 'block' in path...")
            #     texture_path = os.path.join(mod_assets_path, modname, "textures")
            #     if open_texture_file(modname, texture_path, val, ""):
            #         open_texture_result = True
            #     else:
            #         logger.debug(f"\topen_texture_file() -> returned: {open_texture_result}")
            #
            # # mod_name is the name of the mod script started with.
            # # modname is later defined by the namespace of the model.
            # # so if starting mod is minecraft, ignore. else if the namespace is minecraft, try it
            # if not open_texture_result and mod_name != 'minecraft':
            #     # try as minecraft as mod
            #     logger.debug(f"\tRetrying open_texture_file() as 'minecraft' mod...")
            #     texture_path = os.path.join(mod_assets_path, "minecraft", "textures", "block")
            #     # change the new_jar_path and not the modname
            #     if open_texture_file("minecraft", texture_path, val, ""):
            #         open_texture_result = True
            #     else:
            #         logger.debug(f"\topen_texture_file() as 'minecraft' mod returned 'false'.")
            #
            #         if not open_texture_result:
            #             # try again without block in path
            #             logger.debug(
            #                 f"\tRetrying open_texture_file() as 'minecraft' mod, without 'block' in path...")
            #             texture_path = os.path.join(mod_assets_path, "minecraft", "textures")
            #             if open_texture_file("minecraft", texture_path, val, ""):
            #                 open_texture_result = True
            #             else:
            #                 logger.debug(f"\topen_texture_file() as minecraft mod returned false.")
            #
            # if not open_texture_result and mod_name != 'minecraft':
            #     # try checking mod dependencies for the mod texture and return texture_path where it was found
            #     #TODO all texture should be extracted, but should we keep reference of the jar it was found in or mod?
            #     logger.debug(f"try checking mod dependencies for the mod texture...")
            #     new_texture_path: str = check_dependencies_for_texture(modname, val, texture_path)
            #     if new_texture_path != "":
            #         texture_path = new_texture_path
            #         # shouldnt this try open_texture_file??
            #         logger.debug(f"\ttrying open_texture_file() again with new texture_path: '{new_texture_path}'")
            #         #logger.debug(f"\t\t\tnew texture_path set. texture_path: {new_texture_path}")
            #         if open_texture_file(modname, texture_path, val, ""):
            #             open_texture_result = True
            #         else:
            #             logger.warning(f"\topen_texture_file() false, couldnt open texture from new texture path.")
            #     else:
            #         logger.warning(f"\tcheck_dependencies_for_texture returned empty.")
            #         #exit(1)
            #
            # if not open_texture_result:
            #     logger.error(f"\tTexture at texture_path could not be found. Exiting...")
            #     #TODO exit or just skip??
            #     exit(1)
            # else:
            #out_texture_path = os.path.join(out_textures_path, sb_dir)
            qc(f'$definevariable texture_{tex} "{val}"')
            export_texture(val, texture_path, out_textures_path)

            if tex in textureVars:
                logger.debug(f'  Texture variable "{tex}" trying to be redefined. skipped adding.')
            else:
                textures += [tex]
                textureVars[tex] = val

            logger.debug(f"\ttex: {tex}")
            logger.debug(f"\tval: {val}")
            logger.debug(f"\ttextureVars: {textureVars}")

        qc()

    if 'elements' in jmodel or 'components' in jmodel:
        logger.debug("")
        logger.debug("\tif 'elements' in jmodel or 'components' in jmodel:")
        # 'elements' can be labeled 'components'
        # swap out 'elements' for 'components'
        elements: str = 'elements'
        if 'components' in jmodel:
            elements = 'components'

        qc(f'// JSON "elements"')
        qc(f'$definevariable mesh {model_name.replace(" ", "_").replace(".", "_")}')
        qc()

        model = SMDModel()
        model_textures = []

        #global g_test_bones
        #g_test_bones = parse_geometry()

        # {bone.name:{parent:parent.name, cubes:[int]}, bone.name:{parent:parent.name, cubes:[bone.names]} }
        # bone.name:{parent:parent.name, cubes:[int]}
        #   bone.name:{parent:bone.name, cubes:[bone.names]}
        #   parent = self if no parent
        #   cubes = array of ints(elements_index)
        # bone.name:{parent:parent.name, cubes:[bone.names]}
        #   cubes = array of str bone.name
        #
        # must have test_bones set first
        #smd_anim_test: str = handle_bone_placement(g_test_bones)
        # smd_anim_test: str = ''
        # smd_anim_test += 'time 0\n'
        # for b in g_test_bones:
        #     bone: dict = g_test_bones.get(b)
        #     smd_bone_name = bone.get('name')
        #     smd_bone_parent = bone.get('parent')
        #     smd_bone_origin = Vector(*bone.get('origin'))
        #     smd_rotation: float = 0.0
        #     smd_offset = Vector(0, 0, 0)
        #     if smd_bone_parent != smd_bone_name:
        #         #smd_offset = Vector(*g_test_bones[smd_bone_parent]['origin'])
        #         smd_offset = Vector(0, 0, 1)
        #     else:
        #         smd_rotation = -1.570796
        #
        #     logger.debug(f"smd_offset: {smd_offset}")
        #     smd_bone_origin[0], smd_bone_origin[1], smd_bone_origin[2] = smd_bone_origin[0], smd_bone_origin[2], smd_bone_origin[1]
        #     smd_bone_origin = smd_bone_origin - smd_offset
        #     bone_index = indexOf(g_test_bones, b)
        #     # -3.141592, 1.570796
        #     smd_anim_test += f"{bone_index}  {smd_bone_origin.x} {smd_bone_origin.y} {smd_bone_origin.z}  0 0 {smd_rotation}"
        #     if bone_index != (len(g_test_bones) - 1):
        #         smd_anim_test += "\n"
        # smd_anim_test += '0  0.000000 0.000000 0.000000  0 0 0\n'
        # smd_anim_test += '1  0.000000 0.000000 0.000000  0 0 0\n'
        # smd_anim_test += '2  21.000000 13.500000 0.000000  0 0 0\n'
        # smd_anim_test += '3  -21.000000 13.500000 0.000000  0 0 0'
        # smd_anim_test += 'time 0\n'
        # smd_anim_test += '0  0.000000 0.000000 0.000000  0.000000 0.000000 0.000000\n'
        # smd_anim_test += '1  0.000000 0.000000 0.000000  1.570796 -0.000000 0.000000\n'
        # smd_anim_test += '2  21.000000 13.500000 0.000000  -0.000000 -1.570796 0.000000\n'
        # smd_anim_test += '3  -21.000000 13.500000 0.000000  -0.000000 1.570796 0.000000'

        ''' texture variable found in elements, value of texture. 
            eg: "north": {"uv": [0, 0, 16, 16], "texture": "#front", "cullface": "north"}'''
        elem_count: int = -1
        bone_id: int = -1
        for elem in jmodel[elements]:
            logger.debug("\t\tfor elem in jmodel[elements]:")
            start = Vector(*elem['from'])
            end = Vector(*elem['to'])
            elem_count += 1

            # the "height" of Source engine should be z
            start[0], start[1], start[2] = start[0], start[2], start[1]
            end[0], end[1], end[2] = end[0], end[2], end[1]

            size = end - start
            position = start + size / 2
            position = position * Vector(-1, 1, 1)
            # link may always be 1, bone is -1 if root(bone without parent), 0 for every bone with a parent
            cube = model.add_cube(position, size)

            #bone_id += 1
            bone_name: str = ""
            if 'name' in elem:
                bone_id = get_bone_id_by_name(g_test_bones, elem['name'])
                bone_name = elem['name']

            logger.debug(f'\t\tcube: {size} @ {position}')

            for facename, face in elem['faces'].items():
                #TODO for planes like garland wall deco, make a cube
                # only includes 2 directionsadjust from/to also?
                # or move texture out one pixel
                # eg. if only north/south. from [0,0,16] to [16,16,16] from = [0,0,15.75]?
                texture = face['texture']
                if texture[0] != '#':
                    logger.error("\tExpecting face texture to be a texture variable")
                    raise Exception(
                        '\tException: expecting face texture to be a texture variable')

                # TODO: handle overlay texture, maybe by proxy
                if texture == '#overlay':
                    continue

                if texture == '#missing':
                    logger.warning("\ttexture face #missing, culling...")
                    continue

                texture = texture[1:]
                if texture not in model_textures:
                    model_textures += [texture]

                if texture not in textureVars:
                    logger.warning(
                        f'\ttexture variable "{texture}" was undefined, the model "{model_file}" might be template file')
                    if not args.allow_template:
                        logger.error(f'\tno missing texture was allowed, exiting')
                        exit(1)
                    else:
                        textureVars[texture] = '$missing'

                logger.debug(f'\t\tfacename: {facename} -> {resolve_uv(texture)}')

                rotation = 0
                if 'rotation' in face:
                    rotation = face['rotation']

                uv = None
                if 'uv' not in face:
                    if facename in ['east', 'west']:  # yz plane
                        uv = [start.y, start.z, end.y, end.z]
                    elif facename in ['up', 'down']:  # xy plane
                        uv = [start.x, start.y, end.x, end.y]
                    elif facename in ['south', 'north']:  # xz plane
                        uv = [start.x, start.z, end.x, end.z]
                else:
                    uv = face['uv']

                uv = convert_uv(uv, rotation)

                # we need to rotate "up" and "down" face to match minecraft
                if facename in ['up', 'down']:
                    uv = [uv[3], uv[2], uv[1], uv[0]]

                cube_faces = {
                    "east": 0,
                    "west": 1,
                    "up": 2,
                    "down": 3,
                    "south": 4,
                    "north": 5,
                }
                # will have element count, first start with elem_count = 0
                # if bone_id == -1:
                #     bone_id = get_bone_id(g_test_bones, elem_count)
                #     logger.error(f"cube 'name' null, getting 'bone_id' by get_bone_id()")

                #logger.debug(f"elem_count: {elem_count}, bone_id: {bone_id}")
                # for each face, will need to: if test_bones[0].cubes[0] is_int, add test_bones[0].cubes[0]
                cube.add_face(cube_faces[facename], f'@{texture}', uv, 1, bone_id)

            # now model center at (0, 0, 8), the bottom is on the ground
            cube.translate(Vector(8, -8, 0))

            if 'rotation' in elem:
                #TODO add if rotation axis is x or y
                angle = Vector()
                # get bone by id
                #logger.debug(f"bone_id: {bone_id}")
                #logger.debug(f"g_test_bones.get('bone'): {g_test_bones.get('bone')}")
                #logger.debug(f"g_test_bones.get(bone_name): {g_test_bones.get(bone_name)}")
                #logger.debug("<-----------------------")
                #test_bones: dict = g_test_bones.get(bone_name)
                # if test_bones.get("rotation"):
                #     #TODO
                #     # bone need its rotation set when setting up model in elements
                #     # a rotation of -85(bbq model) will show as 0 angle and animations are backwards because of it
                #     logger.debug("using rotations from bone")
                #     angle[0] = test_bones.get("rotation")[0]
                #     angle[1] = test_bones.get("rotation")[1]
                #     angle[2] = test_bones.get("rotation")[2]
                #     angle[0], angle[1], angle[2] = -angle[0], angle[2], angle[1]

                if 'x' in elem['rotation'] or 'y' in elem['rotation'] or 'z' in elem['rotation']:
                    angle[0] = elem['rotation']['x']
                    angle[1] = elem['rotation']['y']
                    angle[2] = elem['rotation']['z']
                    angle[0], angle[1], angle[2] = -angle[0], angle[2], angle[1]

                else:
                    axis = elem['rotation']['axis']
                    _angle = elem['rotation']['angle']
                    angle = Vector()
                    angle[0 if axis == 'x' else 1 if axis == 'y' else 2] = _angle
                    angle[0], angle[1], angle[2] = -angle[0], angle[2], angle[1]

                if 'origin' in elem['rotation']:
                    origin = Vector(*elem['rotation']['origin'])
                else:
                    origin = Vector(8, 8, 8)

                rescale = False
                if 'rescale' in elem['rotation']:
                    rescale = elem['rotation']['rescale']

                # angle = Vector()
                # angle[0 if axis == 'x' else 1 if axis == 'y' else 2] = _angle
                # angle[0], angle[1], angle[2] = -angle[0], angle[2], angle[1]

                origin[1], origin[2] = origin[2], origin[1]
                origin = origin - Vector(8, 8, 0)

                cube.rotate(angle, origin * Vector(-1, 1, 1), rescale)

        qc(f'// JSON "elements[].faces[].texture" list:')
        # use macro to replace texture variable
        qc(f'$definemacro applytextures \\\\')
        for tex in set(model_textures):
            if tex not in textures:
                undefined_textures += [tex]
            qc(f'$renamematerial "@{tex}" $texture_{tex}$ \\\\')
        qc()

        smd_bones: str = make_smd_bones(g_test_bones)
        smd_anim_test: str = handle_bone_placement(g_test_bones)

        logger.debug(f'\tNew smd: {model_file}')
        with open(os.path.join(out_models_path, model_file.replace(" ", "_").replace(".", "_") + '.smd'), 'w', encoding='utf-8') as f:
            smd = LineBuilder()
            smd('version 1')
            smd('nodes')
            #smd('000 "Cube" -1')
            smd(smd_bones)
            smd('end')
            smd('skeleton')
            #smd('time 0')
            #smd(model.to_smdbones(), end='')
            smd(smd_anim_test)
            smd('end')
            smd('triangles')
            smd(model.to_smd(), end='')
            smd('end')
            f.write(str(smd))

        #for group in jmodel[groups].items():

        # if 'children' in group:
        #     children: str = 'children'
        #     children_list: list[str] = groups[children]
        #     for child in children_list:
        #         if 'name' in child:
        #             name: str = 'name'
        #             logger.debug(f"name: {name}")
        #pass

    # if 'undefined_textures' contains any textures not in 'textures', add it to 'require_textures'
    require_textures = []
    logger.debug(f"\tundefined_textures: '{undefined_textures}'")
    logger.debug(f"\ttextures: '{textures}'")
    for tex in undefined_textures:
        if tex not in textures:
            require_textures += [tex]
            logger.debug(f"\t\ttex: '{tex}' not in require_textures: '{require_textures}', adding tex: '{tex}'")

    # in the case of anvil, template would be the real model because it contains everything,
    # first json was just the variant with top face. can be replaced with textures like cracked
    # will still compile as anvil and not template even though thats the qc name.
    real_model = len(require_textures) == 0 and len(textures) > 0

    # # return qc string section to add to qc
    # anim_sequences: str = '$staticprop'
    # if is_geo:
    #     if parse_animation(g_test_bones):
    #         anim_sequences = f'$include "anim_sequences.qci"'

    # if not real_model write qc include
    # parent_file containing everything we need to build it, not just variant textures.
    # but should we use variant textures? probably yes in the case of drawer/bamboo, base json being bamboo style
    # so should the texture be overwritten instead of skipping(where assert is)?
    qc_ext = '.qc' if real_model else '.qci'
    fp = os.path.join(out_models_path, model_file.replace(" ", "_").replace(".", "_") + qc_ext)
    os.makedirs(os.path.dirname(fp), exist_ok=True)

    with open(fp, 'w', encoding='utf-8') as f:
        logger.debug(f"\twith open(fp, 'w', encoding='utf-8') as f:")
        if real_model:
            mdl_path = os.path.join(mod_name, sb_dir, model_subfolders, model_file)
            f.write(QC_HEADER.format(model_file=mdl_path.replace("\\", "/").replace(" ", "_").replace(".", "_"), anim_sequences=anim_sequences))
            logger.debug(f"\t\tWriting QC_HEADER...")
        else:
            logger.debug(f"\t\treal_model false, QC_HEADER will not be written.")

        logger.debug(f"\t\tWriting QC contents...")
        f.write(str(qc))

        if real_model:
            f.write(QC_FOOTER)
            logger.debug(f"\t\tWriting QC_FOOTER...")
        else:
            logger.debug(f"\t\treal_model false, QC_FOOTER will not be written.")

    logger.debug(f"\treturn require_textures: List[str] = '{require_textures}'")
    logger.debug("<---end of 'parse_json(models_path, model_file)'")
    logger.debug("")
    global full_qc_path
    full_qc_path = fp
    global json_model
    json_model = model_file
    return require_textures

def parse_model(models_path: str, model_file: str) -> List[str]:
    """
    Parse model json text file
    Parameters:
        models_path (str): model's path to model json file.
        model_file (str): model's json text file as inputted.
    """
    model_file = model_file.replace(" ", "_").replace(".", "_")
    logger.debug("")
    logger.debug(f'parse_model(models_path, model_file)')
    logger.debug(f"\tmodels_path: '{models_path}'")
    logger.debug(f"\tmodel_file: '{model_file}')")
    # TODO: handle namespace

    new_path: str = os.path.join(models_path, model_file + '.json')
    logger.debug(f"\tnew_path: {new_path}")
    jmodel = open_model_file(new_path, "")

    if jmodel is None:
        logger.warning(f"\t'jmodel' still equals None, checking dependencies.")

        dependency: str
        dependencies: list[str] = getDependencies()

        # TODO check our modId (bibliocraft) first!!, then search all dependencies
        #  cant unless puling it from the texture path, not accurate, find a better way
        logger.debug("")
        logger.debug("*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*")
        logger.debug(f"for dependency in dependencies:")
        for dependency in dependencies:
            # look through all jars for a texture at path
            if dependency != dependencies[0]:
                mod_jar: str = get_mod_jar(dependency)
                #logger.debug(f"[parse_model] dependency: {mod_jar}")
                #logger.warning(f"[parse_model] get_mod_jar(dependency): {mod_jar}")
                logger.debug(f"Checking for dependency: {dependency} in '{mod_name}' dependencies...")
                logger.debug(f"get_mod_jar(dependency): {mod_jar}")
                # TODO can we skip trying to extract and just check if its in the jar?
                #new_assets_path: str = new_path.split("assets\\assets\\")[1]
                #sub_dirs = os.path.dirname(val) + "/"
                new_assets_path: str = ("assets\\" + new_path.split("assets\\assets\\")[1]).replace("\\", "/")
                logger.debug(f"new_assets_path: {new_assets_path}")
                if not check_in_jar(mod_jar, new_assets_path):
                    logger.debug(f"path: {new_assets_path} not in jar: {mod_jar}")
                else:
                    jmodel = open_model_file(new_path, mod_jar)
                    if jmodel is not None:
                        logger.debug(f"\t[JSON_Path]:{new_path}")
                        break

        if jmodel is None:
            logger.warning(f"\t'jmodel' still equals None. Trying 'minecraft' as mod.")
            if mod_name in new_path:
                new_path = new_path.replace(mod_name, "minecraft")
                jmodel = open_model_file(new_path, "")
                if jmodel is None:
                    logger.error(
                        f"\t'jmodel' still equals None, extraction failed or file not found. Path: '{new_path}'")
                    exit(1)
                else:
                    logger.debug(f"\t[JSON_Path]:{new_path}")
            else:
                logger.warning(f"\tmod_name: '{mod_name}' not in path: '{new_path}'")
                logger.error(f"\t'jmodel' equals None, extraction failed or file not found. Path: '{new_path}'")
                exit(1)
    else:
        logger.debug(f"\t[JSON_Path]:{new_path}")

    idx: int = model_file.find('/')
    model_name = model_file[idx + 1:]
    logger.debug(f"\tmodel_name = model_file[idx + 1:]: '{model_file[idx + 1:]}'")

    logger.debug(f'\tCreating new qc: {model_file}')
    qc = LineBuilder()

    textures = []
    ''' key for texture, eg: side '''
    undefined_textures = []
    ''' any texture(value in key:value) starting with # pointing to what tex to reference. 
        textures not defined in model_file? '''

    if 'model' in jmodel:
        #model: str = jmodel['model']
        if '.obj' in jmodel['model']:
            logger.error("\tmodel requires an .obj model and not currently supported.")
            exit(1)

    if 'textures' in jmodel:
        logger.debug("")
        logger.debug(f"if 'textures' in jmodel:")
        qc('// JSON "textures":')

        modname: str = ""
        for tex, val in jmodel['textures'].items():
            logger.debug(f"for tex, val in jmodel['textures'].items():")
            logger.debug(f"\ttex: {tex}")
            logger.debug(f"\tval: {val}")
            if ':' in val:
                logger.debug(f"if ':' in val:")
                # split into two, return second part
                # val = eg: another_furniture:block/drawer/birch_front
                #val = val.split(':')
                modname = val.split(':')[0]
                val = val.split(':')[1]
                logger.debug(f"\tmodname: '{modname}', val: '{val}'")

            if val[0] == '#':
                logger.debug(f"if val[0] == '#'")
                # imported texture
                undefined_textures += [val[1:]]
                logger.debug(f"\tundefined_textures += {[val[1:]]}")
                # add variable for where texture is to be used, in qc
                qc(f'$definevariable texture_{tex} $texture_{val[1:]}$')
            else:
                logger.debug(f"\telse:")
                # real texture
                # maybe need to redefine how the texture is retrieved,
                # if is block/drawer/texture, else do as proceeded
                # val = block/drawer/birch_front

                # split val into subdirectories and file name
                val: str = val.replace('block/', '')

                #open_texture_file(modname, val) # try extracting if not exists
                logger.debug(f"\tval: {val}")
                sub_dirs: str = ""
                if "/" in val:
                    logger.debug(f"if '/' in val:")
                    sub_dirs = os.path.dirname(val) + "/"
                    #sub_dirs = val.replace(new_val, "")
                    new_val: str = val.split("/")[-1]  # alternator last occurrence
                    #sub_dirs = val.replace(new_val, "") # alternator/alternator being replaced leaving '/'
                    val = new_val
                    logger.debug(f"\tval: '{val}'")
                    logger.debug(f"\tsub_dirs: '{sub_dirs}'")

                # assets_path = "...\Minecraft MDL to Source MDL\assets_other\assets"
                logger.debug(f"\tval: '{val}', tex:'{tex}'")  # eg val: front, tex: birch_front
                #qc(f'$definevariable texture_{tex} "{val}"')

                #texture_path = os.path.join(assets_path, f'{modname}/textures/block/{sub_dirs}')
                #texture_path = os.path.join(assets_path, modname, "textures", "block", sub_dirs)

                val = os.path.join(sub_dirs, val)
                # if texture didnt have namespace: at beginning, this will be empty
                # fallback to mod_name of model,
                # could also be minecraft but check later in export textures
                if modname == "":
                    modname = mod_name
                #texture_path = os.path.join(assets_path, modname, "textures", "block")
                # should path contain block ?
                # "0": "endermanoverhaul:entity/badlands/badlands_enderman",
                # is added here when it doesn't need it and fails to find texture
                # is first time in testing this as been a problem

                texture_path = os.path.join(mod_assets_path, modname, "textures", "block")

                # if texture didnt have namespace: prefixed, then use default mod_name
                # may also be something else? maybe just minecraft?

                open_texture_result: bool = False
                if open_texture_file(modname, texture_path, val, ""):
                    open_texture_result = True

                #open_texture_result: bool = open_texture_file(modname, texture_path, val)
                logger.debug(
                    f"\topen_texture_file() -> returned: {open_texture_result}")  # try extracting if not exists

                if not open_texture_result:
                    # try again without block in path
                    #TODO does this really need checked? does block really sometimes need added?
                    logger.debug(f"\tRetrying open_texture_file() without 'block' in path...")
                    texture_path = os.path.join(mod_assets_path, modname, "textures")
                    if open_texture_file(modname, texture_path, val, ""):
                        open_texture_result = True
                    else:
                        logger.debug(f"\topen_texture_file() -> returned: {open_texture_result}")

                # mod_name is the name of the mod script started with.
                # modname is later defined by the namespace of the model.
                # so if starting mod is minecraft, ignore. else if the namespace is minecraft, try it
                if not open_texture_result and mod_name != 'minecraft':
                    # try as minecraft as mod
                    logger.debug(f"\tRetrying open_texture_file() as 'minecraft' mod...")
                    texture_path = os.path.join(mod_assets_path, "minecraft", "textures", "block")
                    # change the new_jar_path and not the modname
                    if open_texture_file("minecraft", texture_path, val, ""):
                        open_texture_result = True
                    else:
                        logger.debug(f"\topen_texture_file() as 'minecraft' mod returned 'false'.")

                        if not open_texture_result:
                            # try again without block in path
                            logger.debug(
                                f"\tRetrying open_texture_file() as 'minecraft' mod, without 'block' in path...")
                            texture_path = os.path.join(mod_assets_path, "minecraft", "textures")
                            if open_texture_file("minecraft", texture_path, val, ""):
                                open_texture_result = True
                            else:
                                logger.debug(f"\topen_texture_file() as minecraft mod returned false.")

                if not open_texture_result and mod_name != 'minecraft':
                    # try checking mod dependencies for the mod texture and return texture_path where it was found
                    #TODO all texture should be extracted, but should we keep reference of the jar it was found in or mod?
                    logger.debug(f"try checking mod dependencies for the mod texture...")
                    new_texture_path: str = check_dependencies_for_texture(modname, val, texture_path)
                    if new_texture_path != "":
                        texture_path = new_texture_path
                        # shouldnt this try open_texture_file??
                        logger.debug(f"\ttrying open_texture_file() again with new texture_path: '{new_texture_path}'")
                        #logger.debug(f"\t\t\tnew texture_path set. texture_path: {new_texture_path}")
                        if open_texture_file(modname, texture_path, val, ""):
                            open_texture_result = True
                        else:
                            logger.warning(f"\topen_texture_file() false, couldnt open texture from new texture path.")
                    else:
                        logger.warning(f"\tcheck_dependencies_for_texture returned empty.")
                        #exit(1)

                if not open_texture_result:
                    logger.error(f"\tTexture at texture_path could not be found. Exiting...")
                    #TODO exit or just skip??
                    exit(1)
                else:
                    #out_texture_path = os.path.join(out_textures_path, sb_dir)
                    qc(f'$definevariable texture_{tex} "{val}"')

                    logger.debug(f"\tbefore export_texture() function")
                    logger.debug(f"\tsub_dirs: '{sub_dirs}'")
                    logger.debug(f"\ttextureVars: '{textureVars}'")
                    #logger.debug(f"\tval: '{val}'")
                    #logger.debug(f"\ttextures_path: '{textures_path}'")
                    #logger.debug(f"\ttexture_path: '{texture_path}'")
                    #logger.debug(f"\tout_textures_path: '{out_textures_path}'")

                    # if texture cant be found in 'textures_path', skipped. should anything else need to happen?
                    # maybe not insert into textures and texturesVars???
                    # in the case of acacia_1_tucked, texture from parent file '"bottom": "another_furniture:block/chair/oak_bottom"'
                    # is excluded because it would overwrite a tex already defined
                    #open_texture_file(textures_path, val) # try extracting if not exists
                    export_texture(val, texture_path, out_textures_path)

            # going though model file, textures added. going through parent_file, textures with same name will redefine
            # them. need to skip adding it. This only seems to happen in custom mod models.json
            logger.debug(f"\ttextureVars: {textureVars}")
            logger.debug(f"\ttrying assert tex not in textureVars")
            #assert tex not in textureVars, f'Texture variable "{tex}" redefined.'

            if tex in textureVars:
                logger.debug(f'  Texture variable "{tex}" trying to be redefined. skipped adding.')
            else:
                textures += [tex]
                textureVars[tex] = val

            logger.debug(f"\ttex: {tex}")
            logger.debug(f"\tval: {val}")
            logger.debug(f"\ttextureVars: {textureVars}")

        qc()

    if 'parent' in jmodel:
        logger.debug("")
        logger.debug("if 'parent' in jmodel:")

        # if parent: is "minecraft:block/block", skip?
        # nothing in there we need at the moment
        logger.debug(f"\tjmodel['parent']: {jmodel['parent']}")
        logger.debug(f"\tparent_exclude_list: {g_parent_exclude_list}")
        parent: str = jmodel['parent']
        logger.debug(f"\tparent as var str: {parent}")
        if check_in_list(parent, g_parent_exclude_list):

            logger.debug(f"\tskipping model, contains minecraft:block. parent: '{jmodel['parent']}'")
        else:
            # in parent file, if parent file's parent file is minecraft mod,
            # and we are using a secondary mod, wont be able to find minecraft assets in mod
            # would need path to mc assets
            logger.debug(f"\tparent_file: {jmodel['parent']}")
            modname: str
            parent: str = jmodel['parent']
            if ":" in parent:
                modname = parent.split(":")[0]
            else:
                modname = mod_name

            #parent_file: str = jmodel['parent'].replace('block/', '').replace(f'{mod_name}:', '')
            parent_file: str = parent.replace('block/', '').replace(f'{modname}:', '')
            logger.debug(f"\tparent_file before split: {parent_file}")

            # split subdirectories from parent_file name
            sub_dirs: str = ""
            if '/' in parent_file:
                logger.debug(f"\tif '/' in parent_file:")
                new_parent_file: str = parent_file.split("/")[-1]
                sub_dirs = os.path.dirname(parent_file) + "/"
                #sub_dirs = parent_file.replace(new_parent_file, "") # bad, if parent file in path twice, it removes both
                parent_file = new_parent_file
                logger.debug(f"\tparent_file: {parent_file}")

            #models_path = os.path.join(assets_path, f'{mod_name}/models/block/{sub_dirs}')
            # TODO is this anything to do with why potted cactus textures are wrong?
            #  potted cactus has no variant -> parent model
            models_path = os.path.join(mod_assets_path, modname, "models", "block", sub_dirs)
            #models_path = os.path.join(mod_assets_path, mod_name, "models", "block", sub_dirs)
            logger.debug(f"\t\tmodels_path: '{models_path}'")
            logger.debug(f"\t\tparent_file: '{parent_file}'")

            # set all textures in anything other then the main model json to undefined_textures?
            logger.debug("\t\tundefined_texture += parse_model(models_path, parent_file)")
            # go through parent_file the same way looking for more textures and elements to convert
            # require_textures would return into undefined_textures
            undefined_textures += parse_model(models_path, parent_file)

            idx = parent_file.find('/')
            qc(f'// JSON "parent":')
            qc(f'$include "{parent_file[idx + 1:]}.qci"')
            qc()

    if 'children' in jmodel:
        # in example mekanism:block/factory/smelting/active/advanced.json
        # indicates other parts of the model, 'children', following textures,
        # "children":{"base":{"parent":"mekanism:block/factory/smelting/base"},
        # "front_led":{"parent":"mekanism:block/factory/front_led/advanced"}}
        # would probably need to cycle through these using as parent.
        # Should be built into qc. Need to check if there is a specific way
        # of including other parts into the qc or just add to the qci.
        # probably would need to get the first elements value, where parent is the key
        # then run the same as parent from above
        pass

    if 'groups' in jmodel:
        pass

    if 'elements' in jmodel or 'components' in jmodel:
        logger.debug("")
        logger.debug("\tif 'elements' in jmodel or 'components' in jmodel:")
        # 'elements' can be labeled 'components'
        # swap out 'elements' for 'components'
        elements: str = 'elements'
        if 'components' in jmodel:
            elements = 'components'

        qc(f'// JSON "elements"')
        qc(f'$definevariable mesh {model_name.replace(" ", "_").replace(".", "_")}')
        qc()

        model = SMDModel()
        model_textures = []

        ''' texture variable found in elements, value of texture. 
            eg: "north": {"uv": [0, 0, 16, 16], "texture": "#front", "cullface": "north"}'''

        for elem in jmodel[elements]:
            logger.debug("\t\tfor elem in jmodel[elements]:")
            start = Vector(*elem['from'])
            end = Vector(*elem['to'])

            # the "height" of Source engine should be z
            start[0], start[1], start[2] = start[0], start[2], start[1]
            end[0], end[1], end[2] = end[0], end[2], end[1]

            size = end - start
            position = start + size / 2
            position = position * Vector(-1, 1, 1)
            cube = model.add_cube(position, size)

            logger.debug(f'\t\tcube: {size} @ {position}')

            for facename, face in elem['faces'].items():
                #TODO for planes like garland wall deco, make a cube
                # only includes 2 directionsadjust from/to also?
                # or move texture out one pixel
                # eg. if only north/south. from [0,0,16] to [16,16,16] from = [0,0,15.75]?
                texture = face['texture']
                if texture[0] != '#':
                    logger.error("\tExpecting face texture to be a texture variable")
                    raise Exception(
                        '\tException: expecting face texture to be a texture variable')

                # TODO: handle overlay texture, maybe by proxy
                if texture == '#overlay':
                    continue

                if texture == '#missing':
                    logger.warning("\ttexture face #missing, culling...")
                    continue

                texture = texture[1:]
                if texture not in model_textures:
                    model_textures += [texture]

                if texture not in textureVars:
                    logger.warning(
                        f'\ttexture variable "{texture}" was undefined, the model "{model_file}" might be template file')
                    if not args.allow_template:
                        logger.error(f'\tno missing texture was allowed, exiting')
                        exit(1)
                    else:
                        textureVars[texture] = '$missing'

                logger.debug(f'\t\tfacename: {facename} -> {resolve_uv(texture)}')

                rotation = 0
                if 'rotation' in face:
                    rotation = face['rotation']

                uv = None
                if 'uv' not in face:
                    if facename in ['east', 'west']:  # yz plane
                        uv = [start.y, start.z, end.y, end.z]
                    elif facename in ['up', 'down']:  # xy plane
                        uv = [start.x, start.y, end.x, end.y]
                    elif facename in ['south', 'north']:  # xz plane
                        uv = [start.x, start.z, end.x, end.z]
                else:
                    uv = face['uv']

                uv = convert_uv(uv, rotation)

                # we need to rotate "up" and "down" face to match minecraft
                if facename in ['up', 'down']:
                    uv = [uv[3], uv[2], uv[1], uv[0]]

                cube_faces = {
                    "east": 0,
                    "west": 1,
                    "up": 2,
                    "down": 3,
                    "south": 4,
                    "north": 5,
                }

                cube.add_face(cube_faces[facename], f'@{texture}', uv, 1, -1)

            # now model center at (0, 0, 8), the bottom is on the ground
            cube.translate(Vector(8, -8, 0))

            if 'rotation' in elem:
                #TODO add if rotation axis is x or y
                angle = Vector()
                if 'x' in elem['rotation'] or 'y' in elem['rotation'] or 'z' in elem['rotation']:
                    angle[0] = elem['rotation']['x']
                    angle[1] = elem['rotation']['y']
                    angle[2] = elem['rotation']['z']
                    angle[0], angle[1], angle[2] = -angle[0], angle[2], angle[1]

                else:
                    axis = elem['rotation']['axis']
                    _angle = elem['rotation']['angle']
                    angle = Vector()
                    angle[0 if axis == 'x' else 1 if axis == 'y' else 2] = _angle
                    angle[0], angle[1], angle[2] = -angle[0], angle[2], angle[1]

                if 'origin' in elem['rotation']:
                    origin = Vector(*elem['rotation']['origin'])
                else:
                    origin = Vector(8, 8, 8)

                rescale = False
                if 'rescale' in elem['rotation']:
                    rescale = elem['rotation']['rescale']

                # angle = Vector()
                # angle[0 if axis == 'x' else 1 if axis == 'y' else 2] = _angle
                # angle[0], angle[1], angle[2] = -angle[0], angle[2], angle[1]

                origin[1], origin[2] = origin[2], origin[1]
                origin = origin - Vector(8, 8, 0)

                cube.rotate(angle, origin * Vector(-1, 1, 1), rescale)

        qc(f'// JSON "elements[].faces[].texture" list:')
        # use macro to replace texture variable
        qc(f'$definemacro applytextures \\\\')
        for tex in set(model_textures):
            if tex not in textures:
                undefined_textures += [tex]
            qc(f'$renamematerial "@{tex}" $texture_{tex}$ \\\\')
        qc()

        logger.debug(f'\tNew smd: {model_file}')
        with open(os.path.join(out_models_path, model_file.replace(" ", "_").replace(".", "_") + '.smd'), 'w', encoding='utf-8') as f:
            smd = LineBuilder()
            smd('version 1')
            smd('nodes')
            smd('000 "Cube" -1')
            smd('end')
            smd('skeleton')
            smd('time 0')
            smd(model.to_smdbones(), end='')
            smd('end')
            smd('triangles')
            smd(model.to_smd(), end='')
            smd('end')
            f.write(str(smd))

    # if 'undefined_textures' contains any textures not in 'textures', add it to 'require_textures'
    require_textures = []
    logger.debug(f"\tundefined_textures: '{undefined_textures}'")
    logger.debug(f"\ttextures: '{textures}'")
    for tex in undefined_textures:
        if tex not in textures:
            require_textures += [tex]
            logger.debug(f"\t\ttex: '{tex}' not in require_textures: '{require_textures}', adding tex: '{tex}'")

    # in the case of anvil, template would be the real model because it contains everything,
    # first json was just the variant with top face. can be replaced with textures like cracked
    # will still compile as anvil and not template even though thats the qc name.
    real_model = len(require_textures) == 0 and len(textures) > 0

    # if not real_model write qc include
    # parent_file containing everything we need to build it, not just variant textures.
    # but should we use variant textures? probably yes in the case of drawer/bamboo, base json being bamboo style
    # so should the texture be overwritten instead of skipping(where assert is)?
    qc_ext = '.qc' if real_model else '.qci'
    fp = os.path.join(out_models_path, model_file.replace(" ", "_").replace(".", "_") + qc_ext)
    os.makedirs(os.path.dirname(fp), exist_ok=True)

    with open(fp, 'w', encoding='utf-8') as f:
        logger.debug(f"\twith open(fp, 'w', encoding='utf-8') as f:")
        if real_model:
            mdl_path = os.path.join(mod_name, sb_dir, model_subfolders, model_file)
            f.write(QC_HEADER.format(
                model_file=mdl_path
                    .replace("\\", "/")
                    .replace(" ", "_")
                    .replace(".", "_"),
                anim_sequences="$staticprop")
            )
            logger.debug(f"\t\tWriting QC_HEADER...")
        else:
            logger.debug(f"\t\treal_model false, QC_HEADER will not be written.")

        logger.debug(f"\t\tWriting QC contents...")
        f.write(str(qc))

        if real_model:
            f.write(QC_FOOTER)
            logger.debug(f"\t\tWriting QC_FOOTER...")
        else:
            logger.debug(f"\t\treal_model false, QC_FOOTER will not be written.")

    logger.debug(f"\treturn require_textures: List[str] = '{require_textures}'")
    logger.debug("<---end of 'parse_json(models_path, model_file)'")
    logger.debug("")
    global full_qc_path
    full_qc_path = fp
    global json_model
    json_model = model_file
    return require_textures

parser = argparse.ArgumentParser(
    description='Convert Minecraft JSON model to Source engine model')
parser.add_argument('model', type=str, help='minecraft model, e.g. "furnace_on", relative to mod_name\\models\\block')
parser.add_argument('--tools', type=str, help='the folder which contains studiomdl.exe',
                    default=r'C:\Program Files (x86)/Steamsteamapps/common/Left 4 Dead 2/bin')
parser.add_argument('--game', type=str, help='the folder which contains gameinfo.txt',
                    default=r'C:\Program Files (x86)\Steam\steamapps\common\Left 4 Dead 2\left4dead2')
parser.add_argument('--assets', type=str, help='path to the mod assets folder with project\'s assets dir')
parser.add_argument('--mcjar', type=str, help='path to the minecraft Jar')
parser.add_argument('--mod_jar', type=str, help='path to the mod Jar')
parser.add_argument('--mod', type=str, help='name of mod eg: minecraft', default='minecraft')
parser.add_argument('--is_geo', type=str, help='true if model is a Bedrock geometry model.', default='false')
parser.add_argument('--geo_path', type=str, help='path to geo entity.json.', default='')
parser.add_argument('-o', '--out', type=str, help='output folder', default='l4d2')
parser.add_argument('--scale', type=int, help='scale in pixels', default=48)
parser.add_argument('--compile-skybox', type=str, default='false', help="compile models for skybox at skybox-scale")
parser.add_argument('--skybox-scale', type=int, help='scale in pixels', default=16)
parser.add_argument('--allow-template', action='store_true')  # usage: args.allow_template

args = parser.parse_args()
tools_path: str = args.tools
game_path: str = args.game
mcjar_path: str = args.mcjar
assets_path: str = args.assets
mod_assets_path: str = os.path.join(args.assets, "assets")
json_model: str = args.model
json_model = json_model.replace(".", "_") # may not want to change this here, may not find the file
out_dir: str = args.out
mod_name: str = args.mod
is_geo: bool = False
geo_path: str = args.geo_path
mod_jar_path: str = args.mod_jar

if args.is_geo == "True":
    is_geo = True

full_qc_path: str = ''
ent_tex: str = ''

#Constants
MOD_JARS: str = "mod_jars.json"

logger.debug("[---------------args-----------------]")
logger.debug(f"tools_path:       '{tools_path}'")
logger.debug(f"game_path:        '{game_path}'")
logger.debug(f"assets_path:      '{assets_path}'")
logger.debug(f"mod_assets_path:  '{mod_assets_path}'")
logger.debug(f"mcjar_path:       '{mcjar_path}'")
logger.debug(f"mod_jar_path:     '{mod_jar_path}'")
logger.debug(f"json_model:       '{json_model}'")
logger.debug(f"is_geo:           '{is_geo}'")
logger.debug(f"geo_path:         '{geo_path}'")
logger.debug(f"out_dir:          '{out_dir}'")
logger.debug(f"mod_name:         '{mod_name}'")
logger.debug(f"scale:            '{args.scale}'")
logger.debug(f"skybox_scale:     '{args.skybox_scale}'")

if not is_geo:
    if mcjar_path == "" or mcjar_path is None:
        logger.error("Error: mc_jar path not set.")
        exit(1)

# adjust for skybox scale if enabled
sb_scale = 1
sb_dir = ""
if args.compile_skybox == "true":
    sb_dir = "skybox"
    sb_scale = args.skybox_scale

pixel_scale = (args.scale / 16) / sb_scale
logger.debug("[------------------------------------]")
logger.debug(f"Converting json_model: {json_model}")

model_subfolders = ""
count = 0
folders = json_model.split("/")
for subfolder in folders:
    if count < len(folders) - 1:
        model_subfolders = os.path.join(model_subfolders, subfolder)
    else:
        json_model = subfolder
    count += 1

logger.debug("Handling subfolders...")
logger.debug(f"len(folders): '{len(folders)}'")
logger.debug(f"model_subfolders: '{model_subfolders}'")
logger.debug(f"folders: '{folders}'")
logger.debug(f"json_model: '{json_model}'")

textures_path: str = os.path.join(mod_assets_path, mod_name, "textures", "block")
model_path: str = os.path.join(mod_assets_path, mod_name, "models", "block", model_subfolders)

out_models_path: str = os.path.join(out_dir, "modelsrc", mod_name, sb_dir, model_subfolders)
out_textures_path: str = os.path.join(out_dir, "materialsrc", mod_name, sb_dir)

# write modelbase_1.qci
model_base_qci: str = os.path.join(out_models_path, 'modelbase_1.qci')

# need to overwrite modelbase_1.qci if scale is updated
#if not os.path.exists(model_base):
logger.debug('creating modelbase_1.qci')

# TODO: dont use global variables
textureDimens = {}
textureVars = {}
g_settings_mod_paths = get_settings_mod_paths()
g_bones_list: list[str] = []
#g_test_bones: dict = {}

g_parent_exclude_list = ["minecraft:block/block", "block/block", "neoforge:block/default"]
'''any parent models in list will be skipped since they are not needed or cant be found.'''

logger.debug("[<-  Start  ->]")

if is_geo:
    parse_geo_model(model_path, json_model)
else:
    parse_model(model_path, json_model)

os.makedirs(out_models_path, exist_ok=True)
with open(model_base_qci, 'w', encoding='utf-8') as f:
    g_path = os.path.join("models", mod_name, sb_dir)
    f.write(QC_MODELBASE.format(path=g_path.replace("\\", "/"), pixel_scale=pixel_scale))

#cmd = f'"{os.path.join(tools_path, "studiomdl.exe")}" -game "{game_path}" -nop4 -nox360 "{os.path.join(out_models_path, json_model)}.qc"'
logger.debug(f"compiling qc at qc_path: '{full_qc_path}'")
cmd = f'"{os.path.join(tools_path, "studiomdl.exe")}" -game "{game_path}" -nop4 -nox360 "{full_qc_path}"'

logger.debug(cmd)
#exit(1)
ok, _, stdout = runcmd(cmd)
if not ok:
    logger.error(f"Log CMD Error: {stdout.decode()}")

std_output: str = stdout.decode().replace("\r\n", "\n")
logger.debug("-> Compiling SMD...")
logger.debug(f"{json_model} Start Build Log\n{std_output}")
logger.debug(f"{json_model} End Build Log")
