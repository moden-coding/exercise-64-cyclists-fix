#!/usr/bin/env python3
"""Tests for the Cyclists pandas assignment."""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.cyclists import cyclists


def _spy_on_method(method_to_spy_on):
    """Wrap an unbound method with a MagicMock that records calls and then
    calls through to the real implementation. Used to check that a method
    like DataFrame.dropna is actually invoked, with which arguments, without
    changing its behavior.
    """
    mock = MagicMock(name="%s method" % getattr(method_to_spy_on, "__name__", "spied"))

    def wrapper(self, *args, **kwargs):
        mock(*args, **kwargs)
        return method_to_spy_on(self, *args, **kwargs)

    wrapper.mock = mock
    return wrapper


class TestCyclists(unittest.TestCase):
    """cyclists() -> a cleaned-up DataFrame of Helsinki cyclist counts."""

    def test_shape(self):
        df = cyclists()
        self.assertEqual(
            df.shape,
            (37128, 21),
            msg="cyclists() should return a DataFrame with shape "
            "(37128, 21). Got shape %r." % (df.shape,),
        )

    def test_columns(self):
        df = cyclists()
        cols = [
            "Baana",
            "Viikintie",
            "Ratapihantie",
            "Lauttasaaren silta pohjoispuoli",
            "Pitkäsilta länsipuoli",
            "Pitkäsilta itäpuoli",
            "Heperian puisto/Ooppera",
            "Munkkiniemi silta pohjoispuoli",
            "Munkkiniemen silta eteläpuoli",
            "Merikannontie",
            "Lauttasaaren silta eteläpuoli",
            "Käpylä, Pohjoisbaana",
            "Kuusisaarentie",
            "Kulosaaren silta po. ",
            "Kulosaaren silta et.",
            "Kaivokatu",
            "Kaisaniemi/Eläintarhanlahti",
            "Huopalahti (asema)",
            "Eteläesplanadi",
            "Auroransilta",
            "Päivämäärä",
        ]
        np.testing.assert_array_equal(
            df.columns,
            cols[::-1],
            err_msg="The DataFrame's columns should match the expected "
            "station names, in the expected (reversed) order.",
        )

    def test_dropna_called_with_how_all(self):
        method = _spy_on_method(pd.core.frame.DataFrame.dropna)
        with patch.object(pd.core.frame.DataFrame, "dropna", new=method):
            cyclists()
            method.mock.assert_called()
            self.assertEqual(
                method.mock.call_count,
                2,
                msg="Expected DataFrame.dropna to be called exactly twice "
                "(once for empty rows, once for empty columns). Got %d "
                "calls." % (method.mock.call_count,),
            )
            for args, kwargs in method.mock.call_args_list:
                self.assertEqual(
                    kwargs.get("how"),
                    "all",
                    msg="Each call to dropna should pass how='all', so only "
                    "rows/columns that are entirely missing get dropped. "
                    "Got kwargs %r." % (kwargs,),
                )


if __name__ == "__main__":
    unittest.main()
