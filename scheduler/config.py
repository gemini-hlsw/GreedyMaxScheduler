# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import os

from omegaconf import OmegaConf

from definitions import ROOT_DIR


class ConfigurationError(Exception):
    """Exception raised for errors in parsing configuration."""

    def __init__(self, config_type: str, value: str):
        """
        Attributes:
            config_type (str): Type of configuration that was being parsed at the moment of the error.
            value (str): Value that causes the error.
        """
        super().__init__(f'Configuration error: {config_type} {value} is invalid.')


path = os.path.join(ROOT_DIR, 'scheduler', 'config.yaml')
config = OmegaConf.load(path)
