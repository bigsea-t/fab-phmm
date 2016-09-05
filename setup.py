"""
Factorized Asymptotic Bayesian Pairwise Hidden Markov Models (FAB-PHMM),
a PHMM with Bayesian model selection capability.
"""


from numpy.distutils.misc_util import get_info
from setuptools import setup, Extension

import fab_phmm

DESCRIPTION = __doc__
VERSION = fab_phmm.__version__
LONG_DESCRIPTION = fab_phmm.__doc__
MAINTAINER = "Taikai Takeda"
MAINTAINER_EMAIL = "297.1951@gmail.com"
LICENSE = "TBD"


setup_options = dict(
    name="fab_phmm",
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,
    license=LICENSE,
    url="https://github.com/hmmlearn/hmmlearn",
    packages=["phmm", "fab_phmm.tests"],
    ext_modules=[
        Extension("fab_phmm.phmmc", ["fab_phmm/phmmc.c"],
                  extra_compile_args=["-O3"],
                  **get_info("npymath"))
    ],
)


if __name__ == "__main__":
    setup(**setup_options)