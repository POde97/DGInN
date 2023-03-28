from setuptools import setup

setup(
    name='DGInN',
    version='1.0',    
    description='Graph Attention Network for Cell Embedding on single cell technologies',
    url='https://github.com/POde97/Cell2CellMatch',
    author='Paolo Odello',
    author_email='paoloodeo.o@gmail.com',
    license='MIT license',
    packages=['DGInN'],
    install_requires=['scanpy==1.9.3',
                     'networkx==2.6.3']
)
