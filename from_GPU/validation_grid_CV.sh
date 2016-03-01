#!/bin/sh

#  validation_timit.sh
#  
#
#  Created by Enea Ceolini on 15/12/15.
#

grid="3"
norm="0"
model="2"
gamma="0.005 0.05 0.1 0.15 0.2 0.25 0.3"
for t in ${grid}
do
    for n in ${norm}
    do
        for m in ${model}
        do
             for g in ${gamma}
             do
            	python validation_grid_CV.py ${t} ${m} ${n} ${g}
             done        
	done
    done
done
