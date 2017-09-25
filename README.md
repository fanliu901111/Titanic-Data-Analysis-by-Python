# Titanic-Data-Analysis-by-Python
## Introduction
## Background information

"RMS Titanic was a British passenger liner that sank in the North Atlantic Ocean in the early morning hours of 15 April 1912, after it collided with an iceberg during her maiden voyage from Southampton to New York City. There were an estimated 2,224 passengers and crew aboard the ship, and more than 1,500 died."   ---"RMS Titanic." Wikipedia. Wikimedia Foundation, 13 July 2017. Web. 14 July 2017.

## Purpose of analysis
This Titanic data contains 891 passengers and crew members' information including Name, gender, survival and etc. 
The purpose of this analysis is to investigate the survival rates depending on several variables. 

One of the reasons that caused this tragedy is that there were not enough life boats and even a lot of the boats were not filled up. In this analysis, I am going to explore that which group of people had a higher survival rate.

## Questions
1. Did women(age>18) had a higher survival rate than men (age>18)?
2. Regardless of gender, did children(age< 18) had a higher survival rate than adults?
3. Did people from Class 1 and 2 had a higher survival rate than people from Class 3?
4. Did women who had children had a higher survival rate than women who did not have children on the board?

## Data Wrangling
### Data Dictionary
(from https://www.kaggle.com/c/titanic)

survival: Surivival(0 = No; 1 = YES)

pclass: Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)

name: Name

sex: Sex

age: Age

sibsp: Number of Siblings/Spouses Aboard

parch: Number of Parents/Children Aboard

ticket: Ticket Number

fare: Passenger Fare

cabin: Cabin

embarked: Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)


### Variable Notes 
pclass: A proxy for socio-economic status (SES)

1st = Upper
2nd = Middle
3rd = Lower

age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5

sibsp: The dataset defines family relations in this way...
Sibling = brother, sister, stepbrother, stepsister
Spouse = husband, wife (mistresses and fiancés were ignored)

parch: The dataset defines family relations in this way...
Parent = mother, father
Child = daughter, son, stepdaughter, stepson
Some children travelled only with a nanny, therefore parch=0 for them.

```
#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
```
```
# Loading the Titanic data#
titanic = pd.read_csv('c:/Users/Fan Liu/Downloads/titanic-data.csv')
```
```python
#To check on the fisrt 5 rows of data to have a basic idea
titanic.head()
```

### Conclusion
According to the analysis, we can see that women did have a higher survival rate than man(Q1). And as children would have a higher survival rate than the adult. Thirdly, people from upper social class definitely had a high survival rate than people from lower social class(Q3). Finally, women with without children had a higher survival rate than those who did not(Q4).

Based on the analysis above, we can see that the women from upper class were most likely to survive since women had a so much higher survival rates than men, and class1's survival rate is the hightest. On the other hand, we can say that man from class3 has the lowest survival rate.

However, Age did not seem to be a major factor based on the analysis.

This data only contains 891 data values out of 2224 for the total number. The data set is less than half the total data, so there might be some unbiased since the data are not fully investigated.

### Notes

I use the mean of 'Age' to replace the missing value which should be noted.
Replacing the missing value with means can make the data more accurate. During the analysis, I need to create variables called adult and child which are all age related.If I keep them with NA, then the 'AGE' variable would only have 714 values. 


The category of 'children' was assumed to be anyone under the age of 18, using today's North American standard for adulthood which was certainly not the case in the 1900s.

### References

https://www.kaggle.com/c/titanic/data

https://en.wikipedia.org/wiki/RMS_Titanic
