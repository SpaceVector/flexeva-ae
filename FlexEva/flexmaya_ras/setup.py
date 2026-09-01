from __future__ import annotations

from pathlib import Path

from setuptools import Extension, setup

import pybind11


ROOT = Path(__file__).resolve().parent

ext_modules = [
    Extension(
        "flexmaya_ras._flexmaya_ras",
        [
            str(ROOT / "cpp" / "flexmaya_core.cpp"),
            str(ROOT / "cpp" / "bindings.cpp"),
        ],
        include_dirs=[
            pybind11.get_include(),
            str(ROOT / "cpp"),
        ],
        language="c++",
        extra_compile_args=["-std=c++17", "-O2", "-Wall", "-Wextra"],
    )
]


setup(
    name="flexmaya-ras",
    version="0.1.0",
    packages=["flexmaya_ras"],
    package_dir={"": "src"},
    ext_modules=ext_modules,
)
