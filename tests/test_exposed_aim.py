from types import SimpleNamespace
import math
import unittest
import numpy as np
from brain.agent import Agent, Threat


class ExposedAimTests(unittest.TestCase):
    def setUp(self):
        self.a = Agent.__new__(Agent)
        self.a.me = {'aim':[0,0], 'energy':100}
        self.th = Threat(1)
        self.th.pos = np.array([100.,0.,0.])
        self.w = [{'kind':'gun','stats':{'speed':252,'range':230},'ammo':12}]
        self.visible = np.array([False,True,False])
        self.a.world = SimpleNamespace(los_batch=lambda A,B,**kw: self.visible)

    def test_head_peeking_above_cover_changes_the_actual_aim(self):
        aim, fire = self.a.weapon_choice(self.th,self.w,['gun'],[0,0,7],100,False,True)
        self.assertAlmostEqual(aim[1], math.atan2(.5,100))
        self.assertTrue(fire[0])

    def test_torso_remains_preferred_if_visible(self):
        self.visible = np.ones(3,dtype=bool)
        point = self.a._shot_point(np.array([0,0,7]),self.th)
        self.assertAlmostEqual(point[2],4.5)

    def test_blocked_target_cannot_fire_even_with_stale_clear_flag(self):
        self.visible = np.zeros(3,dtype=bool)
        self.assertFalse(self.a._shot_clear(np.array([0,0,7]),self.th))
        _, fire = self.a.weapon_choice(self.th,self.w,['gun'],[0,0,7],100,False,True)
        self.assertEqual(fire,[False,False])


if __name__ == '__main__':
    unittest.main()
