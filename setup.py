from setuptools import setup

setup(name='foundation',
      version='0.1',
      description='Framework for RL and beyond',
      url='https://gitlab.cs.washington.edu/fleeb/foundation',
      author='Felix Leeb',
      author_email='fleeb@uw.edu',
      license='MIT',
      packages=['foundation'],
      install_requires=[
            'numpy',
            'matplotlib',
            'torch',
            # 'tensorflow',
            'gym',
            'OpenCV-Python',
            'tabulate',
            'configargparse',
            'ipdb',
      ],
      zip_safe=False)
