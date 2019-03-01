# Classification of images based on their roation


### Usage

Adaboost

```shell
./orient.py test data/test-data.txt models/adaboost_model.txt adaboost
./orient.py train data/test-data.txt models/adaboost_model.txt adaboost
```

Random Forest and best

```shell
./orient.py train data/test-data.txt models/forest_model.txt forest
./orient.py test data/test-data.txt models/forest_model.txt forest
```
KNN (nearest)

```shell

./orient.py train data/test-data.txt models/nearest_model.txt nearest
./orient.py test data/test-data.txt models/nearest_model.txt nearest
```


### Answer for the questions

1. A description of how you formulated the problem:

#### KNN
KNN is about choose K closest elements and the most shared label, so we calculate the distance, and selected the most common label. We also need to carefully choose K, which would need some experiments.

#### Random forest
Random forest is a group of simple decision trees, form a decision committee.

#### Adaboost
We created 100 decision stumps, and then classified each image to be whether it is at 0 or not 0 and 90 or not 90 etc
So we had run the loop 4 times(for each classification) and used the adaboost algorithm to calculate our alpha values
and 100 weak classifiers.
This was tested on the test file 4 times.  The file classified this into the 4 categories based on the final value of the
summation of product of alphas and  the value of h(value returned by weak classifiers)

 2. a brief description of how your program works:

#### Adaboost
Also Commented in the code.
For selecting a decision stump: we take a pixel from the same quadrant because we expiremented with different
decision stumps and this gave us the highest value of accuracy. Our assumption as to why it works is that whenever the
image is rotated by 90 degrees the first quadrant goes to the second.

 3. A discussion of any problems, assumptions, simplifications, and/or design decisions you made


##### Adaboost
Like mentioned above we used the same quadrant for getting a decision stump
We also observed that the accuracy was marginally better when we sampled the pixel for each decision stump from the
same colour

We also preprocessed the data trying to normalise it, but it gave low accuracy, therefore we assume it lost a lot of
features when we normalise the data along the columns


4. introduce how you implement each classifier

In knn, we simply compute manhattan between our test and every training sample, sort them and extract the first k train samples that most close to our test.
Let them vote, the most voted result is our predict result.

In the Random forest, we first implement a basic decision tree using the Gini impurity criterion, although a single tree doesn't give me a good result, a forest can. So we then grow one hundred trees which will form a voting committee, similarly, let them vote, the most voted result is our predict result.

In Ada Boost, we implement a simple decision stump, in one approach we randomly pick pixel in an image sample and classify, in another approached we pick one from the first quadrant as explained above

5. present neatly-organized tables or graphs showing classification accuracies and running times as a function of the parameters you choose.

| Algorithm 	| Train 	| Test 	| Accuracy 	|
|---------------	|-------	|----------------	|----------	|
| KNN 	| null 	| 10min 	| 70.31% 	|
| Random Forest 	| 20min 	| less than 1min 	| 68.93% 	|
| AdaBoost 	| 2min 	| less than 1min 	| 55.47% 	|

6. Which classiers and which parameters would you recommend to a potential client? How does performance vary depending on the training dataset size, i.e. if you use just a fraction of the training data?

For the knn, we tested the k parameter from 3 to 100, k larger than 20 doesn't increase accuracy so we pick k as 20.
![KNN plotting result](data/WX20181209-100624.png)

For the random forest, we tested the number of trees in 10, 100, 100, a forest that has more than 100 trees already gives me a pretty good result and more than 100 seems to make no difference, so we pick 100.

For AdaBoost when the training set was reduced to 500, we saw that the dip in accuracy wasn't too much. At 500 the accuracy was 48%, at 2000 it was around 51%, at 3000 it was again close to 50%. For 5000 training images, the accuracy shot closer to the final accuracy of the model 55%. And then it was observed that the accuracy remained constant for the data.

7. Show a few sample images that were classified correctly and incorrectly. Do you see any patterns to the errors?

We found that some photos that ​mis-classified​ by our algorithm ​was​ ​actually​ ​difficult​ ​to​ ​be identifyied ​by human beings, they can either be rahter blurry or the direction is not clear.

List of wrong (Adaboost):
1. ![predicted : 90 , actual : 270](https://c2.staticflickr.com/6/5327/9995085083_caaedd981c.jpg "Predicted: 90, Actual: 270")
2. ![predicted: 90 , actual : 180](https://c2.staticflickr.com/6/5498/9944682215_0ca008f3b0.jpg "predicted: 90 , actual : 180")
3. ![predicted : 0 , actual : 90](https://c2.staticflickr.com/6/5508/9694305901_782c62dfdb.jpg "predicted : 0 , actual : 90")

List of correctly predicted (Adaboost):
1. ![Angle 0](https://c2.staticflickr.com/6/5467/9760490031_5509d5779f.jpg "Angle 0")
2. ![Angle 90](https://c2.staticflickr.com/6/5468/9646375952_6dc31aa001.jpg "Angle 90")
3. ![Angle 270](https://c2.staticflickr.com/8/7294/9623930399_525ddf3a3b.jpg "Angle 270")

