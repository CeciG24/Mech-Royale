import json
import queue
import subprocess
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from bot import Bot


class EntryPointTests(unittest.TestCase):
    def test_ready_arrives_without_closing_stdin(self):
        proc = subprocess.Popen([sys.executable,'-u','bot.py'],stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE,stderr=subprocess.PIPE,bufsize=0)
        responses = queue.Queue()
        thread = threading.Thread(target=lambda: responses.put(proc.stdout.readline()),daemon=True)
        thread.start()
        try:
            hello = {'type':'hello','protocol':'mechfight/1','config':{},
                     'map':{'dir':'/__missing_test_map__'}}
            proc.stdin.write((json.dumps(hello)+'\n').encode())
            proc.stdin.flush()
            ready = json.loads(responses.get(timeout=5))
            self.assertEqual(ready['type'],'ready')
            proc.stdin.write(b'{"type":"end"}\n')
            proc.stdin.close()
            proc.wait(timeout=3)
            self.assertEqual(proc.returncode,0)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=3)
            if not proc.stdin.closed:
                proc.stdin.close()
            proc.stdout.close()
            proc.stderr.close()
            thread.join(timeout=1)

    def test_map_failure_preserves_fallback(self):
        def fail(hello):
            raise FileNotFoundError('missing test map')
        with patch.dict(sys.modules,{'brain.agent':SimpleNamespace(Agent=fail)}), patch('mechfight.log'):
            bot = Bot({'config':{}})
        self.assertIsNone(bot.agent)
        self.assertEqual(bot.act_errors,0)

    def test_failed_decision_gets_a_fallback_action(self):
        class Broken:
            def __init__(self,hello):
                pass
            def act(self,obs):
                raise ValueError('test failure')
        class Fallback:
            def __init__(self,hello):
                pass
            def act(self,obs):
                return {'move':[1,0]}
        with patch.dict(sys.modules,{'brain.agent':SimpleNamespace(Agent=Broken),
                                     'brain.simple':SimpleNamespace(SimpleBot=Fallback)}), patch('mechfight.log'):
            bot = Bot({'config':{}})
            self.assertEqual(bot.act({'tick':2})['move'],[1,0])
            self.assertEqual(bot.act_errors,1)


if __name__ == '__main__':
    unittest.main()
