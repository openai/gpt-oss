from importlib import import_module as _im

# Load the compiled extension (gpt_oss.metal._metal)
_ext = _im(f"{__name__}._metal")
globals().update({k: v for k, v in _ext.__dict__.items() if not k.startswith("_")})
del _im, _ext
