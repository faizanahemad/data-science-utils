# data-science-utils
Lots of useful functions over Pandas and Python Numpy for Data Science

### Installation

`pip install --upgrade --upgrade-strategy only-if-needed https://github.com/faizanahemad/data-science-utils/tarball/master`

### Usage

Import the following for use
```python

from data_science_utils import dataframe as df_utils
from data_science_utils import models as model_utils
from data_science_utils import plots as plot_utils
from data_science_utils import preprocessing as pp_utils
from data_science_utils.dataframe import column as column_utils
from data_science_utils import misc as misc
from data_science_utils import nlp as nlp_utils
from data_science_utils.models.IdentityScaler import IdentityScaler

```

Reloading Modules on Fly
```python

import importlib
importlib.reload(misc)
importlib.reload(nlp_utils)
importlib.reload(pp_utils)

```

### References or Resources
- [find-the-column-name-which-has-the-maximum-value-for-each-row](https://stackoverflow.com/questions/29919306/find-the-column-name-which-has-the-maximum-value-for-each-row)
- For setting debug points and debugging [read this about Ipython Tracer](http://kawahara.ca/how-to-debug-a-jupyter-ipython-notebook/)
- [Counting Co-occurences](https://stackoverflow.com/questions/42272311/python-co-occurrence-of-two-items-in-different-lists)


# LICENSE

MIT License

Copyright (c) 2019 Faizan Ahemad

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
