# -*- coding: utf-8 -*-
from setuptools import setup

package_dir = \
{'': 'src'}

packages = \
[
    'fedbmr'
]

package_data = \
{'': ['*']}

install_requires = \
['jax',
 'numpyro',
 'optax',
 'seaborn',
 'matplotlib',
 'arviz']

classifiers = \
[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'Topic :: Scientific/Engineering',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.10'
]

setup_kwargs = {
    'name': 'fedbmr',
    'version': '0.0.1',
    'description': 'Probabilistic inference for models of behaviour',
    'long_description': None,
    'author': 'Dimitrije Markovic',
    'author_email': 'dimitrije.markovic@tu-dresden.de',
    'url': 'https://github.com/gaia-os/fedbmr',
    'package_dir': package_dir,
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.10',
}

setup(**setup_kwargs)

