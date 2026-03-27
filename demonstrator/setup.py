from setuptools import setup, find_packages
from setuptools.command.build_py import build_py
import os, shutil, subprocess

class BuildAccelerator(build_py):
    def run(self):
        subprocess.check_call(["make", "build"], cwd="accelerator/csrc", env=os.environ)
        shutil.copy2("accelerator/csrc/build/libaccelerator.so",
                     "accelerator/libaccelerator.so")
        super().run()

setup(
    name="accelerator",
    version="0.1.0",
    packages=find_packages(),
    package_data={"accelerator": ["libaccelerator.so"]},
    cmdclass={"build_py": BuildAccelerator},
)
