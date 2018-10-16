from setuptools import setup, find_packages

setup(name       = 'nenupy',
    packages     = find_packages(),
    include_package_data=True,
    install_requires=['numpy', 'scipy', 'astropy', 'matplotlib'],
    python_requires='>=3.5',
    scripts      = ['bin/nenuplot', 'bin/nenusim'],
    version      = '0.3.5',
    description  = 'NenuFAR Python package',
    url          = 'https://github.com/AlanLoh/nenupy.git',
    author       = 'Alan Loh',
    author_email = 'alan.loh@obspm.fr',
    license      = 'MIT',
    zip_safe     = False)

# make the package:
# python3 setup.py sdist bdist_wheel
# upload it:
# python3 -m twine upload dist/*version*

# Release:
# git tag -a v0.1 -m "annotation for this release"
# git push origin --tags