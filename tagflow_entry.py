#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Hardened TagFlow launcher.

The original :mod:`TagFlow` module remains the GUI implementation.  This
launcher imports it without starting the event loop, installs the reliability
upgrades from :mod:`tagflow_core.integration`, and then invokes its ``main``
function.  Keeping the integration in a small wrapper avoids duplicating the
large legacy GUI module while making the changes independently testable.
"""

from __future__ import annotations

import TagFlow as app

from tagflow_core.integration import install


def main() -> None:
    install(app)
    app.main()


if __name__ == "__main__":
    main()
