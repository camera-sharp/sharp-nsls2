The multi-gpu version of the e1 demo running on a single node.

1. Generate frames and create the e1.cxi file for the sharp-nsls2 program.

./datacxi.py

2. Reconstruct the image and probe using the MPI version of sharp-nsls2.

mpirun -np 2 ./sharp_nsls2.py

3. Show results

./show.py



