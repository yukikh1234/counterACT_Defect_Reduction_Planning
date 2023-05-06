# CounterACT: Counterfactual ACTtion Rule mining Approach

## RQ results of CounterACT

+ To get the results that answer the research questions in the paper, first
run runexp.py to get measurement scores for all planners

+ Then run the specific rqx.py to get results for the corresponding RQ. 

+ A sample result for all 3 RQs is placed under results directory. 

## What is CounterACT?
CounterACT proposes `maintainable` and `achievable` plans to each
individual file within the project.
The following is an example illustrating how CounterACT's plans can be applied
by practitioners. 

#Example

## Step 1: Select the actionable features 
Hedge function will report the features wth the highest variance in the historical data. 
We select the TOP-M features to change in our plans. M is user specifed. In our study we used 5.
 

## Step 2: Action Rule mining
Action Rule mining algorithm will generate Action rules from
the project between `previous` release and `current` release. Such
results will be used to generate plans according to the historical records. 
CounterACT only give plans that happended to `actionable` features.


## Step 3: Generate plans
CounterACT recommends plans by using the action rules mined from historical data


