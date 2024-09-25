# RE⫶TR: Referring Expression Generation with Transformers

To train: 
- Copy ```configuration_template.py``` into a new file ```configuration.py``` and add custom settings
- Run ```main.py``` (arguments as specified in the file can be used to override settings on the fly)

To generate expressions:
- Run ```inference/run_inference.py``` with the ```--checkpoint``` argument pointing to the checkpoint file (other arguments as specified in the file can be used to override settings on the fly)
