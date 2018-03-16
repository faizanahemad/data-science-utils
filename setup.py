from setuptools import setup

setup(name='data_science_utils',
      version='0.1',
      description='Utils for use in python with pandas and numpy',
      url='https://github.com/faizanahemad/data-science-utils',
      author='Faizan Ahemad',
      author_email='fahemad3@gmail.com',
      license='MIT',
      packages=['data_science_utils'],
      install_requires=[
          'numpy','pandas',
      ],
      keywords=['Pandas','numpy','data-science','IPython', 'Jupyter'],
      test_suite='nose.collector',
      tests_require=['nose'],
      zip_safe=False)