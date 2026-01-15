---
title: Error Codes
---

# 🚨 Error Codes

## Error Handling

Pydantic will raise a `ValidationError` whenever it finds an error in the data it's validating.

```python
from pydantic import ValidationError

from infer-ci import MyClass


try:
    _my_object = MyClass(
        config={
            "min_length": 0,
            "max_length": "three",
            "min_value": True,
            "max_value": [1, 2, 3],
            "threshold": 2.0,
        }
    )
except ValidationError as err:
    print(err)
```

The error message will look like this:

```txt
4 validation errors for MyClassConfigPM
min_length
  Input should be greater than or equal to 1 [type=greater_than_equal, input_value=0, input_type=int]
    For further information visit https://errors.pydantic.dev/2.10/v/greater_than_equal
max_length
  Input should be a valid integer, unable to parse string as an integer [type=int_parsing, input_value='three', input_type=str]
    For further information visit https://errors.pydantic.dev/2.10/v/int_parsing
max_value
  Input should be a valid number [type=float_type, input_value=[1, 2, 3], input_type=list]
    For further information visit https://errors.pydantic.dev/2.10/v/float_type
threshold
  Input should be less than or equal to 1 [type=less_than_equal, input_value=2.0, input_type=float]
    For further information visit https://errors.pydantic.dev/2.10/v/less_than_equal
```
