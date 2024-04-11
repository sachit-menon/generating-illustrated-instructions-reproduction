## Training

We use the `accelerate` framework with DeepSpeed to train models. 

By default, models will save in the DeepSpeed format. We provide a convenience script to convert to a standard `.bin` instead in `weight_conversion.sh`. Example usage:
```
bash trainer/scripts/weight_conversion.sh /default/
```

Config details can be found in the appropriate yaml files; major thank you to https://github.com/yuvalkirstain/PickScore which this repo is based on, and where further details on training miscellany can be found.
