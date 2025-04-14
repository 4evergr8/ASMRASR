from a_transcribe import transcribe
from b_slicejson import slicejson
from c_translate import translate
from d_subtitle import subtitle
from getdependency import getdependency
from getconfig import get_config
import importlib.util
from preprocess import preprocess


def main():
    while True:
        if importlib.util.find_spec('whisperx') is not None:
            break
        else:
            getdependency()

    config = get_config()
    preprocess(config)
    transcribe(config)
    slicejson(config)
    translate(config)
    subtitle(config)

if __name__ == "__main__":
    main()