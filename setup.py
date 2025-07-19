from setuptools import setup

setup(
    name='html2obsidian',
    version='0.4.0',
    py_modules=['convert_html', 'test_convert_html'],
    python_requires='>=3.9',
    install_requires=[
        'lxml==5.1.0',
    ],
    extras_require={
        'dev': [
            'pytest>=8.2.0,<9.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'html2obsidian = convert_html:main',
        ],
    },
)
