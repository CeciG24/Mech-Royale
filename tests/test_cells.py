from types import SimpleNamespace
import unittest
import numpy as np
from brain.agent import Agent
from brain.nav import NavGrid


class CellTests(unittest.TestCase):
    def test_vector_queries_match_navigation_grid(self):
        agent = Agent.__new__(Agent)
        nav = NavGrid.__new__(NavGrid)
        nav.w = SimpleNamespace(minx=-100, miny=-200, step=1)
        nav.k, nav.NX, nav.NY = 4, 50, 40
        agent.nav = nav
        # Includes cell boundaries, ties, negative map coordinates and clipping.
        xs = np.arange(-110, 120, .25)
        ys = xs - 100
        jj, ii = agent._cells(xs, ys)
        expected = np.array([nav.cell(x, y) for x, y in zip(xs, ys)])
        np.testing.assert_array_equal(jj, expected[:, 0])
        np.testing.assert_array_equal(ii, expected[:, 1])


if __name__ == '__main__':
    unittest.main()
