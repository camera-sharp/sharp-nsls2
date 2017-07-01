The single-GPU demo that reconstructs the image and probe from simulated frames.
The demo uses data from the data/d1 directory.

1. Generate frames and create the e1.cxi file for the sharp-nsls2 program.
Intermediate results are shown in data2cxi.ipynb

./datacxi.py

2. Reconstruct the image and probe using the sharp-nsls2 algorithm.
You can monitor intermediate results with sharp_nsls2.ipynb

./sharp_nsls2.py

3. Show results

./show.py



