# New Experiments for the thesis paper
## That is a first need to rerun Huang experiments with our data to have more reliable results
Things to do:
- Run their algs with my data (that I have for sure listed in the paper).
Only no context but both binary and soft (just run actually i will only put soft)
    - ~~RNN spectra + joint + discrim~~ _DONE_
    - ~~TIMIT~~ _DONE_
    - ~~TSP~~ _DONE_
- Run NMF (from matlab impl) on my data:
    - ~~Different window sizes (512-1024) (STFT points) --DONE [! I'll only use 512 to keep it constant across models]~~
    - ~~Different number of basis vectors (10-20-30-50-100) --DONE [! Took 100 which overall seems to give the best results!]~~
    - ~~Binary and soft masks~~ _DONE_ **Only  care about soft mask**

**Probably would be nice to address the problem to do MM MF and FF**: just comment, no time to redo everything


# From Reviews
List of stuff to fix:
  - Check Equations of the cost functions
  - Comment on how easy it is to separate similar/different F0s
  - Do something for PESQ/STOI results maybe a comparison --> I rerun with our data, don't wanna put PESQ!   
  - ~~Baseline not reliable~~ _DONE_
  -  **Introduce GRU and explain why not to use LSTM**
  - ~~wrongly stated number for TSP result~~ DONE by rerun
  - Explicitly introduce abbreviation of the abstract
  - In 3.1 should not say ADD the phase
  - in 3.3 don't list the names and discuss the function before the introduction
  - Also in 3.3 check the equations
  - Group references [16][17][18][19] -> [16-19]
  - _Italic_ to enph instead of bold
  - No references in the conclusions
  - better figure RNN
  - Sec 4.1 more details on training data.
  - **make samples**


Considerations about the number of parameters
I estimated the number of parameters depending on the number of units,
they are in both cases quadratic but GRU is of course faster. I also
estimated STOI for TSP depending on the number of parameters in the
network to show that GRU makes efficient use of the parameters
