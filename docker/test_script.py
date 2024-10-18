import numpy as np
import torch
import theseus
import pytorch3d

import pip

def import_or_install(package):
    try:
        __import__(package)
    except ImportError:
        pip.main(['install', package])


if __name__ == '__main__':
    import_or_install('rich')
    from rich import print

    print('Hello World')
    print("Numpy:", np.__version__)
    print("PyTorch:", torch.__version__, f"(GPU available: {torch.cuda.is_available()})")
    print("PyTorch3D:", pytorch3d.__version__)
    print("Theseus:", theseus.__version__)

    print("Script ran successfully")
