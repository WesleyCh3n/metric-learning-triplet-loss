#!/usr/bin/python3
# -*- coding: utf-8 -*-

import importlib.util
import os


def parse_params(path):
    # load parameters
    spec = importlib.util.spec_from_file_location(
        'params', path)
    loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)
    params = loader.params
    return params
