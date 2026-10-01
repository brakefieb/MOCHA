from runners.run_spacontds import run_spacontds

METHOD_REGISTRY = {
    "SpaConTDS": run_spacontds,
    "spacontds": run_spacontds,
}

