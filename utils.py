from pathlib import Path

def getDataPath():
    return str(Path(__file__).resolve().parent.joinpath("dados"))