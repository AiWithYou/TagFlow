#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""TagFlow launcher; --version does not import the Qt GUI."""

from __future__ import annotations

import sys

from tagflow_core.version import __version__


def main() -> None:
    if "--version" in sys.argv[1:]:
        print(f"TagFlow {__version__}")
        return

    import TagFlow as app
    from tagflow_core.integration import install

    app.QApplication.setApplicationVersion(__version__)
    install(app)
    app.main()


if __name__ == "__main__":
    main()
