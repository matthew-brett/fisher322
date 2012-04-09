from distribute_setup import use_setuptools
use_setuptools()

from setuptools import setup


setup(
    name='fisher322',
    version='0.1',
    author='Matthew Brett',
    author_email='matthew.brett@gmail.com',
    packages=['fisher322', 'fisher322.tests'],
    url='',
    license='See LICENSE.txt',
    description='',
    long_description=open('README.txt').read(),
)
