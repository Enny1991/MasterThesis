#!/bin/sh

#  validation_grid.sh
#  
#
#  Created by Enea Ceolini on 15/12/15.
#

grid="1 2 3 4 5 6"
norm="0"
model="0 1 2"
for t in ${grid}
do
    for n in ${norm}
    do
        for m in ${model}
        do
            python validation_grid_fine_tune.py ${t} ${m} ${n}
        done
    done
done
