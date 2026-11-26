"""CLI entry point that forwards directly to the self-contained skill script."""

import importlib.util
import os


def _load_engine():
    # Support both editable/source checkout and installed package layouts
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(pkg_dir, "_engine.py"),
        os.path.join(os.path.dirname(pkg_dir), "ai-audio-clean-master", "scripts", "clean_analog_master.py"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            spec = importlib.util.spec_from_file_location("deai_master_engine", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    raise FileNotFoundError(f"Could not locate DSP engine in {candidates}")


engine = _load_engine()
process_track = engine.process_track
main = engine.main

if __name__ == "__main__":
    main()
