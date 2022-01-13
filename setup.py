from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='compython',
    version='0.0.3',
    packages=['src'],
    url='https://github.com/sth-v/compython.git',
    license='MIT',
    author='Andrew Astakhov',
    author_email='aw.astakhov@gmail.com',
    description='some geometry pack',
    long_description=long_description,
    long_description_content_type="text/markdown"
)
