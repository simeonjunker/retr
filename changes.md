Changed: 

```
catr
|-- datasets
|   |-- coco.py ->  return unique images for evaluation
|
|-- eval_utils
|   |-- decode.py   ->  NEW
|                       greedy decoding scheme (similar to predict.py)
|
|-- configuration.py    ->  changed coco path
|
|-- eval.py ->  NEW
|               evaluate the model over the val dataset (similar to predict.py)
|
|-- .gitignore  ->  added checkpoint.pth
```