#Indoor/Outdoor classifier
Can predict whether the device is indoors or outdoors given the following vector:   
```[gpsAccuracyVertical, gpsAccuracyHorizontal, gpsCourse, gpsSpeed]```

## Example
```python
from in_out_classifier import InOutClassifier
  
# build classifier
clf = InOutClassifier()

# test data
# in the form of [gpsAccuracyVertical, gpsAccuracyHorizontal, gpsCourse, gpsSpeed]
indoor_a = [130, 9, -1, -1]
indoor_b = [40, 4, 2.0, -1]
outdoor = [40, 4, 2.0, 1.5]

# make predictions
prediction = clf.predict([indoor_a, indoor_b, outdoor])

# print results
print prediction
```    
## Accuracy   
98.6 percent in test set.    

## Implementation Details
- Algorithm: RandomForest with 5 trees.    

## Questions
Email me at: will@hacstudios.com