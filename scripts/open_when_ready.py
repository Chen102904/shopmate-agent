"""等本地服务真正起来之后，再打开浏览器。

为什么需要这个：

    `start http://localhost:8000` 是立刻执行的，
    但 uvicorn 还要加载检索模型，第一次启动要 10-20 秒。
    浏览器抢在前面打开，用户看到的是「无法访问此网站」——
    很容易被当成服务起不来。

    这个脚本就干一件事：轮询 /health，等服务能应答了再打开浏览器。

设计取舍：

    - 最多等 90 秒。超时就不开了，用户自己刷新——不能让一个
      辅助脚本永远挂着。
    - 任何异常都静默退出。这个脚本失败不该影响服务本身，
      它只是省了一次手动点链接。
    - 设了 SHOPMATE_NO_BROWSER=1 就只探活不开浏览器，
      方便自动化测试。
"""

import os
import sys
import time
import urllib.error
import urllib.request
import webbrowser


HEALTH_URL = "http://127.0.0.1:8000/health"
PAGE_URL = "http://localhost:8000"

# 轮询间隔：0.5 秒足够及时，又不会把单核 CPU 占满。
POLL_INTERVAL = 0.5

# 上限 180 次 × 0.5 秒 = 90 秒
MAX_ATTEMPTS = 180


def wait_until_ready() -> bool:
    """轮询 /health，返回服务是否就绪。"""

    for _ in range(MAX_ATTEMPTS):
        try:
            with urllib.request.urlopen(HEALTH_URL, timeout=2) as resp:
                if resp.status == 200:
                    return True

        except urllib.error.URLError:
            # 还没起来，或者连不上：正常，继续等。
            pass

        except Exception:
            # 其他异常（比如响应解析问题）不该让等待中断。
            pass

        time.sleep(POLL_INTERVAL)

    return False


def main() -> int:
    if not wait_until_ready():
        return 0

    if os.getenv("SHOPMATE_NO_BROWSER", "").strip() == "1":
        return 0

    try:
        webbrowser.open(PAGE_URL)
    except Exception:
        # 打不开浏览器不影响服务，用户自己敲地址也行。
        pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
