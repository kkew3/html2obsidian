from setuptools import setup

setup(
    name='html2obsidian',
    version='0.5.0',
    packages=['html2obsidian'],
    package_data={
        'html2obsidian': ['xsltml_2.1.2/*.xsl'],
    },
    python_requires='>=3.9',
    install_requires=[
        'lxml>=5.1.0,<=6.0',
    ],
    extras_require={
        'dev': [
            'pytest>=8.2.0,<9.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'html2obsidian = html2obsidian.convert_html:main',
        ],
    },
    scripts=[
        'scripts/diagnose_convergence_issue.py',
        'scripts/inline_xsl_docs.py',
    ],
)
