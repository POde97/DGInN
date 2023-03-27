from setuptools import setup

setup(
    name='DGI-NN',
    version='1.0',    
    description='Graph Attention Network for Cell Embedding on single cell technologies',
    url='https://github.com/POde97/Cell2CellMatch',
    author='Paolo Odello',
    author_email='paoloodeo.o@gmail.com',
    license='MIT license',
    packages=['DGI-NN'],
    install_requires=['scanpy==1.9.3']
)
