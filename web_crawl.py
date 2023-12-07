import requests
import urllib3
import ssl


class CustomHttpAdapter (requests.adapters.HTTPAdapter):
    # "Transport adapter" that allows us to use custom ssl_context.

    def __init__(self, ssl_context=None, **kwargs):
        self.ssl_context = ssl_context
        super().__init__(**kwargs)

    def init_poolmanager(self, connections, maxsize, block=False):
        self.poolmanager = urllib3.poolmanager.PoolManager(
            num_pools=connections, maxsize=maxsize,
            block=block, ssl_context=self.ssl_context)


def get_legacy_session():
    ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
    ctx.options |= 0x4  # OP_LEGACY_SERVER_CONNECT
    session = requests.session()
    session.mount('https://', CustomHttpAdapter(ctx))
    return session


for i in range(100):
    response = get_legacy_session().get(
        "https://selcrs.nsysu.edu.tw/validcode.asp?")

    # 檢查網站是否成功回應
    if response.status_code == 200:
        with open(f"target_dataset/{i}.png", "wb") as img_file:
            img_file.write(response.content)
        print("成功爬取並保存驗證碼圖像")
    else:
        print(
            f"Failed to retrieve website. Status code: {response.status_code}")
