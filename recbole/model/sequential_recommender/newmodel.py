# Thin wrapper to expose NewModel to RecBole's model discovery
try:
    from newmodel import NewModel  # class defined at project root
except Exception as e:
    raise

__all__ = ["NewModel"]


