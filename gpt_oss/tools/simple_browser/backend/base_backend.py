"""
Base backend for the simple browser tool.
"""

from abc import abstractmethod

import chz
from aiohttp import ClientSession

from ..page_contents import PageContents


class BackendError(Exception):
    pass


@chz.chz(typecheck=True)
class Backend:
    source: str = chz.field(doc="Description of the backend source")

    @abstractmethod
    async def search(
        self,
        query: str,
        topn: int,
        session: ClientSession,
    ) -> PageContents:
        pass

    @abstractmethod
    async def fetch(self, url: str, session: ClientSession) -> PageContents:
        pass
