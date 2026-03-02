import collections
import io
import json
import logging
import logging.handlers
import math
import os
import shutil
import struct
import subprocess
import sys
import argparse
import base64
from typing import List, Tuple

from PIL import Image

fmt = '[%(levelname)s] %(asctime)s:\t%(message)s'
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
            "stdin": subprocess.DEVNULL,
        }

        if os.name == "nt":
            popen_params["creationflags"] = 0x08000000

        proc = subprocess.Popen(cmd, **popen_params)
        stdout, stderr = proc.communicate()
        return True, proc.returncode, stdout
    except Exception as err:
        return False, -1, err


reflect_lut = [(i / 255.0)**2.2 for i in range(256)]


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
    def i8(x): return struct.unpack('B', x.to_bytes(1, 'little'))
    def i16(x): return struct.unpack('BB', x.to_bytes(2, 'little'))
    def i32(x): return struct.unpack('BBBB', x.to_bytes(4, 'little'))
    def f32(x): return struct.unpack('BBBB', struct.pack('f', float(x)))

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
        *i16(width), *i16(height),    0x01, 0x23, 0x00, 0x00,
        *i16(len(raws)), *i16(0),     0x00, 0x00, 0x00, 0x00,
        *f32(sX), *f32(sY),
        *f32(sZ),                     0x00, 0x00, 0x00, 0x00,
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
# $cdmaterials "models/{mod_name}/{sb_dir}{model_subfolders}"
QC_MODELBASE = """
$cdmaterials "{path}"
$ambientboost
$scale {pixel_scale}
$staticprop
"""

# $definevariable mdlname "{model_file}"
# $definevariable modname "{mod_name}"
# $definevariable sb_dir  "{sb_dir}"
# $include "modelbase_1.qci"
# $modelname $modname$/$sb_dir$$mdlname$.mdl

QC_HEADER = """// Template header
$definevariable mdlname "{model_file}"
$include "modelbase_1.qci"

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
        return self.x*other.x + self.y*other.y + self.z*other.z

    def __abs__(self):
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

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
        return Vector(self.x, cos*self.y-sin*self.z, sin*self.y+cos*self.z)

    def rotate_y(self, angle):
        rad = math.radians(angle)
        cos = math.cos(rad)
        sin = math.sin(rad)
        return Vector(cos*self.x + sin*self.z, self.y, -sin*self.x + cos*self.z)

    def rotate_z(self, angle):
        rad = math.radians(angle)
        cos = math.cos(rad)
        sin = math.sin(rad)
        return Vector(cos*self.x - sin*self.y, sin*self.x + cos*self.y, self.z)

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

    def to_smd(self, normal: Vector) -> str:
        # <int|Parent bone> <float|PosX PosY PosZ> <normal|NormX NormY NormZ> <normal|U V> [ignores]
        return f'0  {self.pos}  {normal}  {self.uv.x:.5f} {self.uv.y:.5f}'


class Face:
    def __init__(self, texture: str, vertices: List[Vertex], norm: Vector):
        self.tex = texture
        self.vert = vertices
        self.norm = norm

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
        builder += vert[0].to_smd(self.norm) + '\n'
        builder += vert[1].to_smd(self.norm) + '\n'
        builder += vert[2].to_smd(self.norm) + '\n'
        builder += self.tex + '\n'
        builder += vert[1].to_smd(self.norm) + '\n'
        builder += vert[3].to_smd(self.norm) + '\n'
        builder += vert[2].to_smd(self.norm) + '\n'
        return builder


class Cube:
    def __init__(self, position: Vector, size: Vector):
        self.size = size
        self.position = position
        self.faces = []

    def add_face(self, face: int, texture: str, uv: List[Vector]):
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
        face = Face(texture, vertices, norm)
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


def export_texture(texture: str, texture_dir: str, out_dirs: str) -> None:
    logger.debug(f'->export_texture(texture="{texture}", texture_dir, out_dir)')

    out = os.path.normpath(os.path.join(out_dirs, texture))
    if texture == '$missing':
        im = Image.open(io.BytesIO(base64.b64decode(missing_texture)))
    else:
        source = os.path.normpath(os.path.join(texture_dir, texture + '.png'))
        if not os.path.exists(source):
            logger.warning(f'Texture file does not exist: {source}, skipping...')
            return
            #exit(1)

    im = Image.open(source)
    if im.format != 'PNG':
        logger.warning(f'Source texture is not PNG: {texture}')

    if im.mode != 'RGB' and im.mode != 'RGBA':
        im = im.convert('RGBA')

    meta = os.path.normpath(os.path.join(texture_dir, texture + '.png.mcmeta'))
    if os.path.exists(meta):
        with open(meta) as f:
            animation = json.load(f)['animation']

        # TODO:
        if 'frames' in animation:
            logger.warning(
                f'texture {texture} has "animation.frames", but is currently not supported')

        # minecraft animation 20 fps, it might cause some lag in csgo
        if 'frametime' in animation:
            frametime = 20 / animation['frametime']
        else:
            frametime = 20

        # slow it down or fps will drop when player up close
        frametime /= 2

        frame_size = im.width
        assert im.height % frame_size == 0, f'invalid animated texture: {texture}'

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

    # sb_dir = ""
    # new_sb_dir = ""
    # if args.compile_skybox == "true":
    # #     sb_dir = "skybox/"
    #     new_sb_dir = sb_dir + "/"
    #cdmaterials = f'models/{mod_name}/{new_sb_dir}'

    logger.debug("----------------------------------------------------------------------------")
    cdmaterials = os.path.join("models", mod_name)
    if sb_dir != "":
        cdmaterials = os.path.join("models", mod_name, sb_dir)

    logger.debug(f"cdmaterials: '{cdmaterials}'")
    with open(out + '.vmt', 'w') as f:
        f.write(VMT_MODELS_TEMPLATE.format(
            cdmaterials=cdmaterials, texture_file=texture, surfaceprop='', proxy=proxy))

    game_material = os.path.join(game_path, 'materials', cdmaterials, texture)
    logger.debug(f"game_path: '{game_path}'")
    logger.debug(f"game_material: '{game_material}'")
    logger.debug(f"game material path: {os.path.dirname(game_material)}")

    os.makedirs(os.path.dirname(game_material), exist_ok=True)
    shutil.copyfile(out + '.vmt', game_material + '.vmt')
    shutil.copyfile(out + '.vtf', game_material + '.vtf')

    logger.debug("--> end of exporting textures <--")


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

def parse_model(models_path: str, model_file: str) -> List[str]:
    """
    Parse model json text file
    Parameters:
        models_path (str): model's path to model json file.
        model_file (str): model's json text file as inputted.
    """

    # TODO: handle namespace
    # model_file is the name of the parent_file
    # probably should have each texture variant as a different skin index
    # would have to retrieve them all from the folder
    # but how would we know when to do it?
    # then, when selecting a model to compile, what do we select?
    # would have to look at all files related, find parent, see if there are other skins...
    # then might change depending on the mod used
    logger.debug('-> parsing model...')
    logger.debug(f"model_file: {model_file}")
    # taken from json only?
    model_file = model_file.replace(f'{mod_name}:', '')
    logger.debug(f"model_path os joined: {os.path.join(models_path, model_file)}.json")
    with open(os.path.join(models_path, model_file + '.json')) as f:
        jmodel = json.load(f)

    idx: int = model_file.find('/')
    model_name = model_file[idx + 1:]
    logger.debug(f"model_name after model_file[idx + 1:]: {model_file[idx + 1:]}")

    logger.info(f'new qc: {model_file}')
    qc = LineBuilder()

    textures = []
    ''' key for texture, eg: side '''
    undefined_textures = []
    ''' any texture(value in key:value) starting with # pointing to what tex to reference. 
        textures not defined in model_file? '''

    if 'textures' in jmodel:
        qc('// JSON "textures":')
        modname: str = ""
        for tex, val in jmodel['textures'].items():
            if ':' in val:
                # split into two, return second part
                # val = eg: another_furniture:block/drawer/birch_front
                val = val.split(':')
                modname = val[0]
                val = val[1]
                logger.debug(f"if ':' in val -> modname: '{modname}', val: '{val}'")

            if val[0] == '#':
                # imported texture
                undefined_textures += [val[1:]]
                logger.debug(f"undefined_textures += {[val[1:]]}")
                # add variable for where texture is to be used, in qc
                qc(f'$definevariable texture_{tex} $texture_{val[1:]}$')
            else:
                # real texture
                # maybe need to redefine how the texture is retrieved, 
                # if is block/drawer/texture, else do as proceeded
                # val = block/drawer/birch_front

                # split val into subdirectories and file name
                val = val.replace('block/', '')
                logger.debug(f"-=-=-=-=-=-=-=-=val: {val}")
                sub_dirs: str = ""
                if "/" in val:
                    new_val: str = val.split("/")[-1]
                    sub_dirs = val.replace(new_val, "")
                    val = new_val
                    logger.debug(f"val: '{val}', sub_dirs: '{sub_dirs}'")

                # assets_path = "...\Minecraft MDL to Source MDL\assets_other\assets"
                logger.debug(f"val: '{val}', tex:'{tex}'") # eg val: front, tex: birch_front
                #qc(f'$definevariable texture_{tex} "{val}"')

                #texture_path = os.path.join(assets_path, f'{modname}/textures/block/{sub_dirs}')
                #texture_path = os.path.join(assets_path, modname, "textures", "block", sub_dirs)

                val = os.path.join(sub_dirs, val)
                texture_path = os.path.join(assets_path, modname, "textures", "block")
                #out_texture_path = os.path.join(out_textures_path, sb_dir)
                qc(f'$definevariable texture_{tex} "{val}"')

                logger.debug(f"before export_texture() function")
                logger.debug(f"sub_dirs: '{sub_dirs}'")
                logger.debug(f"textureVars: '{textureVars}'")
                logger.debug(f"val: '{val}'")
                logger.debug(f"textures_path: '{textures_path}'")
                logger.debug(f"out_textures_path: '{out_textures_path}'")

                # if texture cant be found in 'textures_path', skipped. should anything else need to happen?
                # maybe not insert into textures and texturesVars???
                # in the case of acacia_1_tucked, texture from parent file '"bottom": "another_furniture:block/chair/oak_bottom"'
                # is excluded because it would overwrite a tex already defined
                export_texture(val, textures_path, out_textures_path)


            # going though model file, textures added. going through parent_file, textures with same name will redefine
            # them. need to skip adding it. This only seems to happen in custom mod models.json
            logger.debug(f"------>textureVars: {textureVars}")
            logger.debug(f"------>trying assert tex not in textureVars")
            #assert tex not in textureVars, f'Texture variable "{tex}" redefined.'

            if tex in textureVars: logger.debug(f'Texture variable "{tex}" trying to be redefined. skipped adding.')
            else:
                textures += [tex]
                textureVars[tex] = val

            logger.debug(f"------>tex: {tex}")
            logger.debug(f"------>val: {val}")
            logger.debug(f"------>textureVars: {textureVars}")

            # if tex not in textureVars:
            #     textureVars[tex] = val
            #     logger.info(f"\"assert tex not in textureVars, Texture variable '{tex}' redefined.\"")
            #     logger.info("Allowing it to pass for now.")
            #     # what's happening is textures are being written to textureVars, then goes to parent json.
            #     # sees front again in textures and overwrites it since front is already defined and cant be added again.
            #     # was giving error texture variable redefined. When going to parent, textures are processed and may change
            #     # should probably not add textures from parent files
            #
            #     # original code
            #     #assert tex not in textureVars, f'Texture variable "{tex}" redefined.'
            #     #textureVars[tex] = val
            # else:
            #     # would have done 'textureVars[tex] = val' if not asserted.
            #     logger.info(f"tex: '{tex}' IS in textureVars: '{textureVars}'")

        qc()

    # if parent is included, should it go and get textures again?
    # skip for now?
    if 'parent' in jmodel:
        logger.debug("-> if 'parent' in jmodel - if statement")

        # if parent: is "minecraft:block/block", skip?
        # nothing in there we need at the moment
        if jmodel['parent'] == "minecraft:block/block":
            logger.debug(f"skipping model, contains minecraft:block. parent: '{jmodel['parent']}'")
        else:
            # in parent file, if parent file's parent file is minecraft mod,
            # and we are using a secondary mod, wont be able to find minecraft assets in mod
            # would need path to mc assets
            logger.debug(f"---------------------------->")
            logger.debug(f"parent_file: {jmodel['parent']}")
            parent_file: str = jmodel['parent'].replace('block/', '').replace(f'{mod_name}:', '')
            logger.debug(f"parent_file before split: {parent_file}")

            # split subdirectories from parent_file name
            sub_dirs: str = ""
            if "/" in parent_file:
                new_parent_file: str = parent_file.split("/")[-1]
                sub_dirs = parent_file.replace(new_parent_file, "")
                parent_file = new_parent_file
                logger.debug(f"parent_file after split: {parent_file}")

            #models_path = os.path.join(assets_path, f'{mod_name}/models/block/{sub_dirs}')
            models_path = os.path.join(assets_path, mod_name, "models", "block", sub_dirs)
            logger.debug(f"models_path: '{models_path}'")
            logger.debug(f"parent_file: '{parent_file}'")

            # set all textures in anything other then the main model json to undefined_textures?
            logger.debug("-> undefined_texture += parse_model(models_path, parent_file)")
            # go through parent_file the same way looking for more textures and elements to convert
            # require_textures would return into undefined_textures
            undefined_textures += parse_model(models_path, parent_file)

            idx = parent_file.find('/')
            qc(f'// JSON "parent":')
            qc(f'$include "{parent_file[idx + 1:]}.qci"')
            qc()


    if 'elements' in jmodel or 'components' in jmodel:
        # 'elements' can be labeled 'components'
        # swap out 'elements' for 'components'
        elements: str = 'elements'
        if 'components' in jmodel:
            elements = 'components'

        logger.debug(f"-> if 'elements' in jmodel - if statement")
        qc(f'// JSON "elements"')
        qc(f'$definevariable mesh {model_name}')
        qc()

        model = SMDModel()
        model_textures = []
        ''' texture variable found in elements, value of texture. 
            eg: "north": {"uv": [0, 0, 16, 16], "texture": "#front", "cullface": "north"}'''

        for elem in jmodel[elements]:
            start = Vector(*elem['from'])
            end = Vector(*elem['to'])

            # the "height" of Source engine should be z
            start[0], start[1], start[2] = start[0], start[2], start[1]
            end[0], end[1], end[2] = end[0], end[2], end[1]

            size = end - start
            position = start + size / 2
            position = (position) * Vector(-1, 1, 1)
            cube = model.add_cube(position, size)

            logger.debug(f'cube: {size} @ {position}')

            for facename, face in elem['faces'].items():
                texture = face['texture']
                if texture[0] != '#':
                    raise Exception(
                        'expect face texture to be a texture variable')

                # TODO: handle overlay texture, maybe by proxy
                if texture == '#overlay':
                    continue

                texture = texture[1:]
                if texture not in model_textures:
                    model_textures += [texture]

                if texture not in textureVars:
                    logger.warning(
                            f'texture variable "{texture}" was undefined, the model {model_file} might be template file')
                    if not args.allow_template:
                        logger.error(f'no missing texture was allowed, exiting')
                        exit(1)
                    else:
                        textureVars[texture] = '$missing'

                logger.debug(f'facename: {facename} -> {resolve_uv(texture)}')

                rotation = 0
                if 'rotation' in face:
                    rotation = face['rotation']

                uv = None
                if 'uv' not in face:
                    if facename in ['east', 'west']:     # yz plane
                        uv = [start.y, start.z, end.y, end.z]
                    elif facename in ['up', 'down']:     # xy plane
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
                    "east":  0,
                    "west":  1,
                    "up":    2,
                    "down":  3,
                    "south": 4,
                    "north": 5,
                }
                cube.add_face(cube_faces[facename], f'@{texture}', uv)

            # now model center at (0, 0, 8), the buttom is on the ground
            cube.translate(Vector(8, -8, 0))

            if 'rotation' in elem:
                axis = elem['rotation']['axis']
                _angle = elem['rotation']['angle']

                if 'origin' in elem['rotation']:
                    origin = Vector(*elem['rotation']['origin'])
                else:
                    origin = Vector(8, 8, 8)

                rescale = False
                if 'rescale' in elem['rotation']:
                    rescale = elem['rotation']['rescale']

                angle = Vector()
                angle[0 if axis == 'x' else 1 if axis == 'y' else 2] = _angle
                angle[0], angle[1], angle[2] = -angle[0], angle[2], angle[1]

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

        logger.info(f'New smd: {model_file}')
        with open(os.path.join(out_models_path, model_file + '.smd'), 'w', encoding='utf-8') as f:
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
    logger.debug(f"undefined_textures: '{undefined_textures}'")
    logger.debug(f"textures: '{textures}'")
    for tex in undefined_textures:
        if tex not in textures:
            require_textures += [tex]
            logger.debug(f"tex: '{tex}' not in require_textures: '{require_textures}', adding tex: '{tex}'")

    # in the case of anvil, template would be the real model because it contains everything,
    # first json was just the variant with top face. can be replaced with textures like cracked
    # will still compile as anvil and not template even though thats the qc name.
    real_model = len(require_textures) == 0 and len(textures) > 0

    # if not a real_model then make it .qci???
    # does real_model mean not the parent_file??
    # parent_file containing everything we need to build it, not just variant textures.
    # but should we use variant textures? probably yes in the case of drawer/bamboo, base json being bamboo style
    # so should the texture be over written instead of skipping(where assert is)?
    qc_ext = '.qc' if real_model else '.qci'
    fp = os.path.join(out_models_path, model_file + qc_ext)
    os.makedirs(os.path.dirname(fp), exist_ok=True)

    # sb_dir = ""
    # if args.compile_skybox == "true":
    #     sb_dir = "skybox/"
    with open(fp, 'w', encoding='utf-8') as f:
        if real_model:
            #f.write(QC_HEADER.format(model_file=model_subfolders + model_file, sb_dir=sb_dir, mod_name=mod_name))
            # $modname$/$sb_dir$$mdlname$.mdl
            mdl_path = os.path.join(mod_name, sb_dir, model_subfolders, model_file)
            f.write(QC_HEADER.format(model_file=mdl_path.replace("\\", "/")))

        f.write(str(qc))

        if real_model:
            f.write(QC_FOOTER)

    logger.debug(f"return require_textures: List[str] = '{require_textures}'")
    logger.debug("-> end of 'parse_json(models_path, model_file)'")
    return require_textures


parser = argparse.ArgumentParser(
    description='Convert Minecraft JSON model to Source engine model')
parser.add_argument('model', type=str, help='minecraft model, e.g. "furnace_on", relative to mod_name\\models\\block')
parser.add_argument('--tools', type=str, help='the folder which contains studiomdl.exe',
                    default=r'C:\Program Files (x86)/Steamsteamapps/common/Left 4 Dead 2/bin')
parser.add_argument('--game', type=str, help='the folder which contains gameinfo.txt',
                    default=r'C:\Program Files (x86)\Steam\steamapps\common\Left 4 Dead 2\left4dead2')
parser.add_argument('--assets', type=str, help='path to the mod assets folder with project\'s assets dir')
parser.add_argument('--mod', type=str, help='name of mod eg: minecraft', default='minecraft')
parser.add_argument('-o', '--out', type=str, help='output folder', default='l4d2')
parser.add_argument('--scale', type=int, help='scale in pixels', default=48)
parser.add_argument('--compile-skybox', type=str, default='false', help="compile models for skybox at skybox-scale")
parser.add_argument('--skybox-scale', type=int, help='scale in pixels', default=16)
parser.add_argument('--allow-template', action='store_true') # usage: args.allow_template
args = parser.parse_args()

tools_path: str = args.tools.replace("\\","/")
game_path: str = args.game.replace("\\","/")
assets_path: str = args.assets.replace("\\","/")
json_model: str = args.model.replace("\\","/")

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
        #model_subfolders += subfolder + "/"
        model_subfolders = os.path.join(model_subfolders, subfolder)
    else:
        json_model = subfolder
    count += 1
# if model_subfolders.endswith("/"):
#     model_subfolders = model_subfolders[len(model_subfolders)-1]

logger.debug(f"len(folders): '{len(folders)}'")
logger.debug(f"model_subfolders: '{model_subfolders}'")
logger.debug(f"folders: '{folders}'")
logger.debug(f"json_model: '{json_model}'")
#print("scale:", pixel_scale)

mod_name: str = args.mod
# if model_subfolders != "":
#     # TODO: textures_path still being used in resolve_uv(). Check if using correct path
#     #textures_path = os.path.join(assets_path, f'{mod_name}/textures/block/{model_subfolders}')
#     #textures_path = os.path.join(assets_path, mod_name, "textures", "block")
#     #model_path = os.path.join(assets_path, f'{mod_name}/models/block/{model_subfolders}')
#     model_path = os.path.join(assets_path, mod_name, "models", "block", model_subfolders)
# else:
#     #textures_path = os.path.join(assets_path, f'{mod_name}/textures/block')
textures_path: str = os.path.join(assets_path, mod_name, "textures", "block")
    #model_path = os.path.join(assets_path, f'{mod_name}/models/block')
#model_path = os.path.join(assets_path, mod_name, "models", "block")
model_path: str = os.path.join(assets_path, mod_name, "models", "block", model_subfolders)

out_dir: str = args.out.replace("\\","/")
our_dir = out_dir#.replace("\\", "/")
logger.debug(f"out path: {out_dir}")
#out_models_path = os.path.join(out_dir, f'modelsrc/{mod_name}/{sb_dir}{model_subfolders}')
out_models_path: str = os.path.join(out_dir, "modelsrc", mod_name, sb_dir, model_subfolders)
#out_textures_path = os.path.join(out_dir, f'materialsrc/{mod_name}/{sb_dir}{model_subfolders}')
out_textures_path: str = os.path.join(out_dir, "materialsrc", mod_name, sb_dir)

# write modelbase_1.qci
model_base: str = os.path.join(out_models_path, 'modelbase_1.qci')

# need to overwrite modelbase_1.qci if scale is updated
#if not os.path.exists(model_base):
# sb_dir = ""
# if args.compile_skybox == "true":
#     sb_dir = "skybox"
logger.debug('creating modelbase_1.qci')

os.makedirs(out_models_path, exist_ok=True)
with open(model_base, 'w', encoding='utf-8') as f:
    # "models/{mod_name}/{sb_dir}{model_subfolders}"

    #print(QC_MODELBASE.format(mod_name=mod_name, model_subfolders=model_subfolders, sb_dir=sb_dir, pixel_scale=pixel_scale), file=f)
    #path = os.path.join("models", mod_name, sb_dir, model_subfolders)
    path = os.path.join("models", mod_name, sb_dir)
    logger.debug(f"------------------------------------> {path.replace("\\","/")}")
    #f.write(QC_MODELBASE.format(mod_name=mod_name, model_subfolders=model_subfolders, sb_dir=sb_dir, pixel_scale=pixel_scale))
    f.write(QC_MODELBASE.format(path=path.replace("\\","/"), pixel_scale=pixel_scale))

# TODO: dont use global variables
textureDimens = {}
textureVars = {}

logger.debug("<-  Start  ->")
parse_model(model_path, json_model)

cmd = f'"{os.path.join(tools_path, "studiomdl.exe")}" -game "{game_path}" -nop4 -nox360 "{os.path.join(out_models_path, json_model)}.qc"'

logger.debug(cmd)
ok, _, stdout = runcmd(cmd)
if not ok:
    logger.error(stdout.decode())

logger.info("-> Compiling SMD...")
std_output: str = stdout.decode().replace("\r\n", "\n")
logger.debug(std_output)
