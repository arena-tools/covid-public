from setuptools import setup, find_packages


setup(name='arenacovid',
      setup_requires=['setuptools-git-version'],
      version_format='{tag}.dev{commitcount}+{gitsha}',
      packages=find_packages(),
      install_requires=['pyarrow', 
                        'pandas==0.24.2',
                        's3fs==0.2.2', 
                        'boto3',
                        'numba',
                        'pymc3',
                        'theano',
                        'matplotlib']
)