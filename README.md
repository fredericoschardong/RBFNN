# RBFNN
A (simple) radial base function neural network (RBFNN) for sklearn


# Programming Exercise 6: Radial Base Function (RBF) Neural Network
## Description
Python (sklearn-based) implementation that explores how different heuristics and parameters impact a RBFNN. 

A brief analysis of the results is [provided in Portuguese](https://github.com/fredericoschardong/RBFNN/raw/main/report%20in%20Portuguese.pdf). It was submited as an assignment of a graduate course named [Connectionist Artificial Intelligence](https://moodle.ufsc.br/mod/assign/view.php?id=2122514) at UFSC, Brazil.

In short, three heuristics (out of my head) are evaluated to find out what is the (least worst) way to calculate sigma, an importat parameter for RBF. Then, the best-performing heuristic is used to test how many neurons are needed in the hidden layer to get a close to 1 f1-score at classifying data from a 2D dataset.
