import kasa
import asyncio
# LED_IP='10.0.0.155'
LED_IP = '10.65.69.82'
# UV_IP='10.0.0.254'
UV_IP = '10.65.69.105'
PLUGS = {"UV":UV_IP,"LED":LED_IP}
class Plug:
    def __init__(self,plug_name):
        '''
        plug_name is a member of the dict above
        '''
        if plug_name not in PLUGS:
            raise RuntimeError(f"Provided name {plug_name} not registered in Plug.py")
        self.plug=kasa.SmartPlug(PLUGS[plug_name])
        self.loop=asyncio.get_event_loop()
    def on(self):
        self.loop.run_until_complete(self.plug.turn_on())
    def off(self):
        self.loop.run_until_complete(self.plug.turn_off())
