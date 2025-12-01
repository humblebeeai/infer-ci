import setuptools

with open('README.md', mode='r', encoding='utf-8') as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as f:
    requirements = f.readlines()

setuptools.setup(
    name='confidenceinterval',
    version='1.0.0',
    author='Humblebee ai',
    author_email='shokulovshohruh.hbai@gmail.com',
    description='Confidence Intervals in python',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/humblebeeai/infer-ci',
    project_urls={
        'Bug Tracker': 'https://github.com/humblebeeai/infer-ci/issues',
    },
    classifiers=[
        'Programming Language :: Python :: 3.10',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    packages=['confidenceinterval'],
    python_requires='>=3.10',
    install_requires=requirements,
    extras_require={
        'detection': [
            'opencv-python-headless',
            'psutil>=6.1.0',
            'py-cpuinfo>=9.0.0',
            'pyyaml>=6.0.0',
            'scipy>=1.14.0',
            'seaborn>=0.13.0',
            'torch',
            'torchaudio',
            'torchvision',
            'tqdm>=4.67.0',
            'ultralytics-thop>=2.0.0',
            'wandb>=0.19.0',
        ]
    })
