import math
import unittest

import numpy as np

from brain.agent import Agent, DEFAULTS, Threat
from brain.aim import yaw_pitch


class SupportFireTests(unittest.TestCase):
    def setUp(self):
        self.agent = Agent.__new__(Agent)
        self.agent.P = dict(DEFAULTS)
        self.agent.pos = np.array([0., 0., 0.])
        self.enemy = Threat(1)
        self.enemy.pos = np.array([100., 0., 0.])
        self.enemy.vel = np.zeros(3)
        self.agent.me = dict(aim=list(yaw_pitch((0, 0, 7), (100, 0, 4.5))),
                             stats=dict(muzzle_height=7), weapons=[
                                 dict(kind='gun', ammo=12, cooldown=0, reload=0,
                                      stats=dict(range=230, speed=252)),
                                 dict(kind='laser', charge=.5)])
        self.target = (self.enemy, 2., True)

    def test_aligned_gun_can_fire_while_laser_charges_or_releases(self):
        for laser_trigger in [False, True]:
            with self.subTest(laser_trigger=laser_trigger):
                fire = [False, laser_trigger]
                self.agent.support_gun(self.target, fire)
                self.assertEqual(fire, [True, laser_trigger])

    def test_gun_does_not_fire_at_unled_moving_target(self):
        self.enemy.vel = np.array([0., 12., 0.])
        fire = [False, True]
        self.agent.support_gun(self.target, fire)
        self.assertEqual(fire, [False, True])

    def test_no_shots_when_quiet_or_obstructed_or_out_of_range(self):
        self.agent.me['weapons'][1]['charge'] = 0
        fire = [False, False]
        self.agent.support_gun(self.target, fire)
        self.assertEqual(fire, [False, False])
        for target in [None, (self.enemy, 2., False)]:
            fire = [False, True]
            self.agent.support_gun(target, fire)
            self.assertEqual(fire, [False, True])
        self.enemy.pos[0] = 250
        fire = [False, True]
        self.agent.support_gun(self.target, fire)
        self.assertEqual(fire, [False, True])

    def test_empty_reloading_and_cooling_gun_stays_off(self):
        gun = self.agent.me['weapons'][0]
        for key, value in [('ammo', 0), ('reload', .3), ('cooldown', .1)]:
            with self.subTest(key=key):
                old = gun[key]
                gun[key] = value
                fire = [False, True]
                self.agent.support_gun(self.target, fire)
                self.assertEqual(fire, [False, True])
                gun[key] = old


if __name__ == '__main__':
    unittest.main()
