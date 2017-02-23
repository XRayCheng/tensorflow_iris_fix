# tensorflow_iris_fix
Since the origin iris sample doesn't work with the new tensorflow(like 1.0, 0.12),
so here is the fix version of that.

the sample page :
https://www.tensorflow.org/get_started/tflearn


Just changed the old parameters with custom input pipe-lines, introduced on this page.
https://www.tensorflow.org/get_started/input_fn

#Tips
1.please get the iris_predict.csv.

2.please make sure you delete the old tmp data, if you have any in the tmp folder.

3.this will be overfitting, if you try to train it twice in a row.
(get [1,1] instead of [1,2])
