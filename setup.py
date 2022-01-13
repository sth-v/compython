import setuptools
import time

lt = time.localtime()
ver = f"{lt[0]}.{lt[1]}.{lt[2]}-{lt[3]}"


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="compython",
    version=ver,
    author="Andrew Astakhov",
    author_email="aw.astakhov@gmail.com",
    description="compython",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sth-v/compython",
    project_urls={
        "Bug Tracker": "https://github.com/sth-v/compython/issues",
    },
    install_requires=["numpy>=1.16.6", "scipy>=1.7.1", "sklearn>=1.0.1"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],

    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.9",
)