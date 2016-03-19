#Indoor/Outdoor classifier
Can predict whether the device is indoors or outdoors given the following vector:   
[gpsAccuracyHor, gpsAccuracyVert, gpsCourse, gpsSpeed]

## Example
```python
from indoor_outdoor_classifier.indoor_outdoor_classifier import InOutClassifier
  
# build classifier
clf = InOutClassifier(verbose=True)

# test data
# in the form of [gpsAccuracyHorizontal, gpsAccuracyVertical, gpsCourse, gpsSpeed]
indoor_a = [130, 9, -1, -1]
indoor_b = [40, 4, 2.0, -1]
outdoor = [40, 4, 2.0, 1.5]

# make predictions
prediction = clf.predict([indoor_a, indoor_b, outdoor])

# print results
print prediction
```