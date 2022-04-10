# StreetNet
AI trained to play [country streaks on GeoGuessr](https://www.geoguessr.com/country-streak). When placed in a random location on Google Street View, this neural network is tasked with predicting which country you are in.

You can test StreetNet by playing a game of GeoGuessr, or by simply dropping into a random location on Google Maps!

##### Screenshot of the StreetNet GUI running while in a game of GeoGuessr, where the location was correctly identified as Finland. 
![StreetNetInFinland](https://user-images.githubusercontent.com/26235672/162554558-a799554d-931c-46f0-bc05-000525339acc.jpg)

## How To Use

To run, simply clone this repo, install the requirements (`requirements.txt` or `cuda_requirements.txt`) and run with python:
```
python streetnet.py model_name [--use-cuda]
```

Currently there are 2 trained models included in this repo: `model_18.pt` and `model_50.pt`. These use the ResNet-18 and the ResNet-50 architectures respectively. `model_50.pt` achieves a higher accuracy, however it can require up to 10.6GB of memory so you may prefer using more light-weight version - `model_18.pt`.

## Training

TODO: Jupyter Notebook and info on dataset and training will be added soon
