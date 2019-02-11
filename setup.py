from setuptools import find_packages, setup

setup(name='data_science_utils',
      version='0.2.18',
      description='Utils for use in python with pandas and numpy',
      url='https://github.com/faizanahemad/data-science-utils',
      author='Faizan Ahemad',
      author_email='fahemad3@gmail.com',
      license='MIT',
      install_requires=[
          'numpy','pandas','beautifulsoup4','fastnumbers','more-itertools',
            'dill','stockstats','pytidylib','seaborn','gensim','nltk','fastnumbers',
            'joblib',
      ],
      keywords=['Pandas','numpy','data-science','IPython', 'Jupyter','ML','Machine Learning'],
      packages=find_packages(),
      test_suite='nose.collector',
      tests_require=['nose'],
      zip_safe=False)
