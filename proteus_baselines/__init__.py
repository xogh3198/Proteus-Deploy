"""Independent GPU adapters for Proteus's singleton comparison baselines.

Not the standalone Stanford INFaaS or native Clipper serving systems.
"""
BASELINES = ("infaas-accuracy", "clipper-ha", "clipper-ht")
SUPPORTED_BASELINES = (*BASELINES, "sommelier")
DISPLAY_NAMES = {"infaas-accuracy": "INFaaS-Accuracy", "clipper-ha": "Clipper-HA", "clipper-ht": "Clipper-HT"}
DISPLAY_NAMES["sommelier"] = "Sommelier (Proteus fixed-placement GPU port)"
