from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='PhaseSelect',
      version='1.1.0',
      description='Phase fields representation learning, ranking syntesibility and classification',
      url='http://github.com/DrewNow/Phase2Vec',
      author='Andrij Vasylenko',
      author_email='and.vasylenko@gmail.com',
      license='MIT',
      packages=['Atom2Vec', 'Models', 'DATA'],
      install_requires=['tensorflow==2.7.0',
                        'numpy==1.22.3',
                        'pandas==1.4.4', 
                        'pymatgen==2022.9.21', 
                        'scikit-learn==1.1.2'],
      python_requires='>=3.7',
      include_package_data=True,
#      entry_points={"console_scripts": ["phase_fields_bo=phase_fields_bo.__main__:run"]},  
      zip_safe=False)
