#!/bin/bash

rm -rf build dist
python3 setup.py sdist bdist_wheel
python3 -m twine upload -r testpypi dist/*
