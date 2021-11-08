from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='Phase2Vec',
      version='1.0.1',
      description='Phase fields representation as vectors with learning their clustering',
      url='http://github.com/DrewNow/Phase2Vec',
      author='Andrij Vasylenko',
      author_email='and.vasylenko@gmail.com',
      license='MIT',
      packages=['Attention', 'DB', 'Atom2Vec'],
      install_requires=['numpy', 'pandas', 'pymatgen'],
      python_requires='>=3.7, <3.9',
      include_package_data=True,
#      entry_points={"console_scripts": ["phase_fields_bo=phase_fields_bo.__main__:run"]},  
      zip_safe=False)
