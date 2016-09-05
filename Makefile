PYTHON = python
CYTHON = cython

all: clean inplace

inplace:
	$(PYTHON) setup.py build_ext --inplace

clean:
	$(PYTHON) setup.py clean

cython:
	find fab_phmm -name "*.pyx" | xargs $(CYTHON)$

