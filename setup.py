from setuptools import setup, find_packages

setup(name       = 'nenupy',
    packages     = find_packages(),
    include_package_data=True,
    version      = '0.1',
    description  = 'NenuFAR Python package',
    url          = 'https://github.com/AlanLoh/nenupy.git',
    author       = 'Alan Loh',
    author_email = 'alan.loh@obspm.fr',
    license      = 'MIT',
    zip_safe     = False)