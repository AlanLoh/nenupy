from setuptools import setup, find_packages
import re
import nenupy

meta_file = open('nenupy/metadata.py').read()
metadata  = dict(re.findall("__([a-z]+)__\s*=\s*'([^']+)'", meta_file))

setup(name       = 'nenupy',
    packages     = find_packages(),
    include_package_data=True,
    install_requires=['numpy', 'scipy', 'astropy', 'matplotlib'],
    python_requires='>=3.5',
    scripts      = ['bin/nenuplot', 'bin/nenusim', 'bin/nenucor', 'bin/nenuinfo', 'bin/nenuplot-gui', 'bin/nenubeam', 'bin/nenuskybeam'],
    version      = nenupy.__version__,
    description  = 'NenuFAR Python package',
    url          = 'https://github.com/AlanLoh/nenupy.git',
    author       = metadata['author'],
    author_email = metadata['email'],
    license      = 'MIT',
    zip_safe     = False)

# make the package:
# python3 setup.py sdist bdist_wheel
# upload it:
# python3 -m twine upload dist/*version*

# Release:
# git tag -a v*version* -m "annotation for this release"
# git push origin --tags

# Documentation
# sphinx-build -b html docs/ docs/_build/

# Update on nancep:
# /usr/local/bin/pip3.5 install nenupy --install-option=--prefix=/cep/lofar/nenupy3 --upgrade