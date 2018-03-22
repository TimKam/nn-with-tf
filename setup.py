import codecs, setuptools

setuptools.setup(
    name='nn-with-tf',
    packages=['src'],
    version='0.0.1',
    description='Simple neural networks with tensorflow',
    author='Timotheus Kampik',
    author_email='tkampik@cs.umu.se',
    url='https://github.com/TimKam/nn-with-tf/',
    platfoxrms=["any"],
    license="MIT",
    zip_safe=False,
    install_requires=[],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Environment :: Plugins",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Topic :: Documentation",
    ],
    long_description=codecs.open("README.rst", "r", "utf-8").read(),
)
