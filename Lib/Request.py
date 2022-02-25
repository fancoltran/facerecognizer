import requests
from config import *


class Request:
    @staticmethod
    def call(api, params):
        try:
            resp = requests.post(url='https://{}/api/{}'.format(DOMAIN, api), json=params, timeout=2)
        except:
            return {"status": "FAIL", "message": "Internet error"}

        return {"status": "SUCCESS", "message": "Success", "response": resp.json()}
