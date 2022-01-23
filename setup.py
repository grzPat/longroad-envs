import os
import sys
import subprocess
from pathlib import Path

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
from subprocess import CalledProcessError


# From https://github.com/pybind/cmake_example/blob/master/setup.py
# Convert distutils Windows platform specifiers to CMake -A arguments
PLAT_TO_CMAKE = {
    "win-amd64": "x64",
}


# A CMakeExtension needs a sourcedir instead of a file list.
# The name must be the _single_ output extension from the CMake build.
# If you need multiple extensions, see scikit-build.
class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])  # "src/pywrapper.cpp"
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def build_extension(self, ext):

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(
            ["cmake", ext.sourcedir], cwd=self.build_temp
        )
        subprocess.check_call(
            ["cmake", "--build", "."], cwd=self.build_temp
        )

        for ext in self.extensions:
            self.move_output(ext)

    def move_output(self, ext):
        build_temp = Path(self.build_temp).resolve()
        dest_path = Path(self.get_ext_fullpath(ext.name)).resolve()
        source_path = build_temp / \
            Path(self.get_ext_filename(ext.name)).parts[-1]
        print(self.get_ext_filename(ext.name))
        print(source_path)
        print(dest_path)

        dest_directory = dest_path.parents[1]
        dest_directory.mkdir(parents=True, exist_ok=True)
        self.copy_file(source_path, dest_path)


kwargs = dict(
    name='longroad',
    version='0.1',
    author='Patrick Grzywok',
    author_email='p.grzywok@tum.de',
    description='longroad environment',
    long_description='',
    ext_modules=[CMakeExtension('longroad.world')],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
    test_suite="tests",
    packages=find_packages(),
    install_requires=['gym', 'numpy', 'pettingzoo']

)

try:
    setup(**kwargs)
except CalledProcessError:
    print('Failed to build extension!')
    del kwargs['ext_modules']
    setup(**kwargs)
