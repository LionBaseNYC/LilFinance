# Utilities

## Setup

* Include the following custom ```import_module()``` function in your script in order to have access to utilities included in this subdirectory:

```
def import_module(name, filepath):
    """
    Function to solve the relative import curse, used in replace of
    'from utils.sentiment_analysis_utils import *' which didn't work.

    @param name: Used to create or access a module object.
    @param filepath: Pathname argument that points to the source file.
    """
    # imp module provides an interface to the mechanisms used to implement the import statement
    import os, imp
    # @param file: The source file, open for reading as text, from the beginning.
    pathname = filepath if os.path.isabs(filepath) else os.path.join(os.path.dirname(__file__), filepath)
    return imp.load_source(name, pathname)

```

* Then, change the ```name``` and ```filepath``` arguments accordingly to load and import any module you want as shown below:

```
sentiment_analysis_utils = import_module(name = "sentiment_analysis_utils",
                                         filepath = "../utils/sentiment_analysis_utils.py")
from sentiment_analysis_utils import *
```

## Notes

* The script ```train_lstm.py``` will relatively take a long time to execute.
