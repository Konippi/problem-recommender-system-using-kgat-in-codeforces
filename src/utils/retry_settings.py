import sys

sys.path.append("../..")

from requests import Session
from requests.adapters import HTTPAdapter, Retry


def setup() -> Session:
    retry = Retry(total=5, backoff_factor=5, status_forcelist=[500, 502, 503, 504, 521, 524])
    session = Session()
    session.mount("http://", HTTPAdapter(max_retries=retry))
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session
