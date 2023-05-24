Below is the WebPPL code for the study. First, Bayesian data analysis was conducted to see whether the probability of choosing risky option was truly affected by belief towards mental illness, frame conditions and disease conditions. To determine the prior distributions, the following was implemented:

~~~~
var logistic = function(x) { 1 / (1+Math.exp(-x))}

var model = function(){
  var w0 = gaussian(0,1)
  var w1 = gaussian(0,1)
  var w2 = gaussian(0,1)
  var w3 = gaussian(0,1)

  return {w0: w0, w1: w1, w2: w2, w3: w3}
  return predictive
}

var options = {method: "MCMC", samples: 5000}
var dist = Infer(options, model)
viz.marginals(dist)


print("Probability of positive w1: " + expectation(marginalize(dist, "w1"), function(p){p > 0}))
print("Expected w1: " + expectation(marginalize(dist, "w1")))

print("Probability of positive w2: " + expectation(marginalize(dist, "w2"), function(p){p > 0}))
print("Expected w2: " + expectation(marginalize(dist, "w2")))

print("Probability of positive w3: " + expectation(marginalize(dist, "w3"), function(p){p > 0}))
print("Expected w3: " + expectation(marginalize(dist, "w3")))
~~~~

The prior distributions can be observed from the figures above. The below code contains the data collected for the current study. Details about the dataset can be found on the submitted report. We can see that the coefficients for mental score and frame appear to be especially prominent, indicating that these variables could be important for the relationship. 

~~~~
// The full dataset is below:
var risk = [0,1,1,1,0,0,0,1,1,1,1,0,1,1,0,1,0,0,1,0,0,0,0,1,0,0,1,1,1,0,0,0,0,1,1,0,1,1,0,0,1,1,1,1,1,0,1,0,0,1,1,1,0,1,0,0,0,1,1,1,1,1,1,1,0,1,0,1,1,1,0,1,1,1,1,1,1,1,0,0,1,1,0,0,1,0,0,1,1,1,1,0,0,0,0,1,0,1,0,1,0,1,0,0,1,1,1,0,1,1,1]

var disease_type = [0,0,0,0,1,0,1,1,1,1,0,1,1,1,1,1,0,0,0,0,1,0,0,1,1,1,1,1,1,1,0,0,1,1,0,0,1,0,0,0,0,1,0,1,1,0,1,1,0,0,0,1,0,0,1,1,0,0,0,0,1,0,1,1,0,1,0,1,0,1,0,0,0,1,0,0,1,1,0,0,0,1,1,1,1,0,0,1,0,1,1,1,1,1,0,1,0,1,1,1,1,0,0,1,0,1,0,0,1,0,1]

var frame = [1,1,1,1,1,0,1,0,0,1,1,1,1,0,1,0,0,0,0,1,1,0,1,0,1,0,0,1,1,0,1,1,0,0,0,1,1,0,1,1,0,0,1,1,1,0,0,1,1,1,1,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,0,0,0,1,1,0,1,1,0,1,0,0,0,1,0,0,0,1,0,1,1,1,1,0,1,0,0,1,0,1,1,0,0,0,1,0,0,1]

var mental_illness_score = [1.571,1.714,1.952,2.000,2.000,2.095,2.143,2.286,2.286,2.286,2.381,2.381,2.381,2.429,2.429,2.524,2.571,2.571,2.571,2.571,2.571,2.62,2.619,2.619,2.619,2.667,2.667,2.667,2.667,2.714,2.762,2.762,2.762,2.762,2.810,2.810,2.810,2.857,2.857,2.857,2.905,2.905,2.952,2.952,2.952,3.000,3.000,3.000,3.048,3.048,3.095,3.143,3.190,3.190,3.190,3.190,3.238,3.238,3.238,3.238,3.238,3.286,3.286,3.286,3.333,3.333,3.381,3.381,3.429,3.429,3.476,3.476,3.476,3.524,3.571,3.571,3.571,3.571,3.619,3.619,3.619,3.619,3.619,3.714,3.714,3.762,3.762,3.762,3.905,3.905,3.905,3.952,3.952,3.952,4.000,4.000,4.048,4.095,4.143,4.143,4.238,4.286,4.286,4.286,4.333,4.333,4.524,4.571,4.714,4.857,4.905]

var subjIDs = _.range(mental_illness_score.length)

// Combining the above dataset
var obsData = map(function(datum) {return _.zipObject(
  ["subjID", "mentalScore" , "riskTaking", "diseaseCondition", "frame"], datum)},
               _.zip(subjIDs, mental_illness_score, risk, disease_type, frame)); 

var logistic = function(x) { 1 / (1+Math.exp(-x))}

var model = function(){
  var w0 = gaussian(0,1)
  var w1 = gaussian(0,1)
  var w2 = gaussian(0,1)
  var w3 = gaussian(0,1)
  
  var obsFn = function(datum){
    observe(Binomial({p: logistic(w0 + w1*datum.mentalScore + w2*datum.diseaseCondition + w3*datum.frame), n: 111}), datum.riskTaking)
  }
  
  mapData({data: obsData}, obsFn)
  
  return {w0: w0, w1: w1, w2: w2, w3: w3}
}

var options = {method: "MCMC", samples: 5000}
var dist = Infer(options, model)
viz.marginals(dist)


print("Probability of positive w1: " + expectation(marginalize(dist, "w1"), function(p){p > 0}))
print("Expected w1: " + expectation(marginalize(dist, "w1")))

print("Probability of positive w2: " + expectation(marginalize(dist, "w2"), function(p){p > 0}))
print("Expected w2: " + expectation(marginalize(dist, "w2")))

print("Probability of positive w3: " + expectation(marginalize(dist, "w3"), function(p){p > 0}))
print("Expected w3: " + expectation(marginalize(dist, "w3")))
~~~~

Moreover, when we remove the effect of belief towards mental illness, the effect of the disease group and frame group become relevant.

~~~~
// The full dataset is below:
var risk = [0,1,1,1,0,0,0,1,1,1,1,0,1,1,0,1,0,0,1,0,0,0,0,1,0,0,1,1,1,0,0,0,0,1,1,0,1,1,0,0,1,1,1,1,1,0,1,0,0,1,1,1,0,1,0,0,0,1,1,1,1,1,1,1,0,1,0,1,1,1,0,1,1,1,1,1,1,1,0,0,1,1,0,0,1,0,0,1,1,1,1,0,0,0,0,1,0,1,0,1,0,1,0,0,1,1,1,0,1,1,1]

var disease_type = [0,0,0,0,1,0,1,1,1,1,0,1,1,1,1,1,0,0,0,0,1,0,0,1,1,1,1,1,1,1,0,0,1,1,0,0,1,0,0,0,0,1,0,1,1,0,1,1,0,0,0,1,0,0,1,1,0,0,0,0,1,0,1,1,0,1,0,1,0,1,0,0,0,1,0,0,1,1,0,0,0,1,1,1,1,0,0,1,0,1,1,1,1,1,0,1,0,1,1,1,1,0,0,1,0,1,0,0,1,0,1]

var frame = [1,1,1,1,1,0,1,0,0,1,1,1,1,0,1,0,0,0,0,1,1,0,1,0,1,0,0,1,1,0,1,1,0,0,0,1,1,0,1,1,0,0,1,1,1,0,0,1,1,1,1,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,0,0,0,1,1,0,1,1,0,1,0,0,0,1,0,0,0,1,0,1,1,1,1,0,1,0,0,1,0,1,1,0,0,0,1,0,0,1]

var mental_illness_score = [1.571,1.714,1.952,2.000,2.000,2.095,2.143,2.286,2.286,2.286,2.381,2.381,2.381,2.429,2.429,2.524,2.571,2.571,2.571,2.571,2.571,2.62,2.619,2.619,2.619,2.667,2.667,2.667,2.667,2.714,2.762,2.762,2.762,2.762,2.810,2.810,2.810,2.857,2.857,2.857,2.905,2.905,2.952,2.952,2.952,3.000,3.000,3.000,3.048,3.048,3.095,3.143,3.190,3.190,3.190,3.190,3.238,3.238,3.238,3.238,3.238,3.286,3.286,3.286,3.333,3.333,3.381,3.381,3.429,3.429,3.476,3.476,3.476,3.524,3.571,3.571,3.571,3.571,3.619,3.619,3.619,3.619,3.619,3.714,3.714,3.762,3.762,3.762,3.905,3.905,3.905,3.952,3.952,3.952,4.000,4.000,4.048,4.095,4.143,4.143,4.238,4.286,4.286,4.286,4.333,4.333,4.524,4.571,4.714,4.857,4.905]

var subjIDs = _.range(mental_illness_score.length)

// Combining the above dataset
var obsData = map(function(datum) {return _.zipObject(
  ["subjID", "mentalScore" , "riskTaking", "diseaseCondition", "frame"], datum)},
               _.zip(subjIDs, mental_illness_score, risk, disease_type, frame)); 

var logistic = function(x) { 1 / (1+Math.exp(-x))}

var model = function(){
  var w0 = gaussian(0,1)
  //var w1 = gaussian(0,1)
  var w2 = gaussian(0,1)
  var w3 = gaussian(0,1)
  
  var obsFn = function(datum){
    observe(Binomial({p: logistic(w0 + /*w1*datum.mentalScore*/ + w2*datum.diseaseCondition + w3*datum.frame), n: 111}), datum.riskTaking)
  }
  
  mapData({data: obsData}, obsFn)
  
  return {w0: w0, /*w1: w1,*/ w2: w2, w3: w3}
}

var options = {method: "MCMC", samples: 5000}
var dist = Infer(options, model)
viz.marginals(dist)


//print("Probability of positive w1: " + expectation(marginalize(dist, "w1"), function(p){p > 0}))
//print("Expected w1: " + expectation(marginalize(dist, "w1")))

print("Probability of positive w2: " + expectation(marginalize(dist, "w2"), function(p){p > 0}))
print("Expected w2: " + expectation(marginalize(dist, "w2")))

print("Probability of positive w3: " + expectation(marginalize(dist, "w3"), function(p){p > 0}))
print("Expected w3: " + expectation(marginalize(dist, "w3")))
~~~~

To capture this relationship, an one-shot soft-max decision agent was implemented. The model appears to be performing similarly to the human participants. 

~~~~
// The variables required can be found below:

var diseaseAssign = ["cognitive", "physical"]
var frameAssing = ["positive", "negative"]
var actions = ["risk", "nonrisk"]

// The utility function is built by 
var utility = function(choice, disease, frame, mental){
  return (frame == "negative" ? 
         (disease == "cognitive" ? (mental > 3.0 ? (choice == "risk" ? 10 : 1) : 
                                    (choice == "risk" ? 5 : 3)) : (choice == "risk" ? 10 : 9)) : 
          (disease == "cognitive" ? (mental > 3.0 ? (choice == "risk" ? 8 : 10) : 
                                     (choice == "risk" ? -1 : 8)) : (choice == "risk" ? 1 : 5)))
}

var alpha = 1

// The softmax agent

var softMaxAgent = function(disease, frame, mental){
  return Infer({
    model(){
      var action = uniformDraw(actions)

      var expectedUtility = function(action){
        return expectation(Infer({
          model(){
            return utility(action, disease, frame, mental)
          }
        }))
      }
      factor(alpha * expectedUtility(action))
      
      return action
    }})
}

// The model can predict what action an agent can take given the conditions
viz(softMaxAgent("cognitive", "negative", 3))
~~~~

A final prototype model for BDA of the cognitive model is below.

~~~~
var diseaseAssign = ["cognitive", "physical"]
var frameAssing = ["positive", "negative"]
var actions = ["risk", "nonrisk"]

// transition function

var transition = function(state){
  var diseaseAssign = flip() ?"cognitive" : "physical"
  
  var frameAssing = flip() ? "positive" : "negative"
  
  return {frameAssing: frameAssing, diseaseAssign: diseaseAssign}
  
}

// The utility function is built by 
var utility = function(choice, disease, frame, mental){
  return (frame == "negative" ? 
         (disease == "cognitive" ? (mental > 3.0 ? (choice == "risk" ? 10 : 1) : 
                                    (choice == "risk" ? 5 : 3)) : (choice == "risk" ? 10 : 9)) : 
          (disease == "cognitive" ? (mental > 3.0 ? (choice == "risk" ? 8 : 10) : 
                                     (choice == "risk" ? -1 : 8)) : (choice == "risk" ? 1 : 5)))
}

// The softmax agent
var softMaxAgent = function(state, mental, alpha){
  return Infer({
    model(){
      var action = uniformDraw(actions)
      var tr = transition(state)
      var frame = tr["frameAssing"]
      var disease = tr["diseaseAssign"]
      
      var expectedUtility = function(action){
        return expectation(Infer({
          model(){return utility(action, disease, frame, mental)
          }
        }))
      }
      factor(alpha * expectedUtility(action))
      
      return action
    }})
}

// The full dataset is below:
var risk = ["risk","risk","risk","nonrisk","nonrisk","risk","risk","risk","risk","risk","nonrisk","risk","nonrisk","nonrisk","risk","risk","risk","risk","nonrisk","nonrisk","risk","risk","risk","risk","risk","nonrisk","risk","risk","nonrisk","nonrisk","nonrisk","nonrisk","nonrisk","nonrisk","nonrisk","nonrisk","nonrisk","nonrisk","risk","risk","risk","risk","risk","risk","risk","risk","nonrisk","nonrisk","nonrisk","nonrisk","nonrisk","nonrisk","nonrisk","nonrisk","nonrisk","nonrisk","nonrisk","risk","risk","risk","risk","risk","risk","risk","risk","risk","risk","risk","nonrisk","nonrisk","nonrisk","nonrisk","nonrisk","nonrisk","risk","risk","risk","risk","risk","risk","risk","risk","risk","risk","risk","risk","risk","risk","nonrisk","nonrisk","nonrisk","nonrisk","nonrisk","nonrisk","nonrisk","nonrisk","nonrisk","nonrisk","nonrisk","nonrisk","risk","risk","risk","risk","risk","risk","risk","risk","risk","risk","risk"
]
var disease_type = ["physical","physical","physical","physical","physical","cognitive","cognitive","cognitive","cognitive","physical","cognitive","cognitive","physical","physical","physical","physical","physical","physical","physical","physical","cognitive","cognitive","cognitive","cognitive","cognitive","cognitive","cognitive","cognitive","physical","physical","physical","physical","physical","physical","physical","physical","physical","physical","physical","physical","physical","physical","physical","physical","physical","physical","physical","physical","physical","physical","physical","physical","physical","physical","physical","physical","physical","physical","physical","physical","physical","physical","physical","physical","physical","physical","physical","physical","cognitive","cognitive","cognitive","cognitive","cognitive","cognitive","cognitive","cognitive","cognitive","cognitive","cognitive","cognitive","cognitive","cognitive","cognitive","cognitive","cognitive","cognitive","cognitive","cognitive","cognitive","cognitive","cognitive","cognitive","cognitive","cognitive","cognitive","cognitive","cognitive","cognitive","cognitive","cognitive","cognitive","cognitive","cognitive","cognitive","cognitive","cognitive","cognitive","cognitive","cognitive","cognitive","cognitive"
]
var frame = ["negative","negative","negative","positive","positive","negative","negative","negative","negative","negative","positive","positive","negative","negative","negative","negative","negative","negative","positive","positive","negative","negative","negative","negative","negative","positive","positive","positive","negative","negative","negative","negative","negative","negative","negative","negative","negative","negative","negative","negative","negative","negative","negative","negative","negative","negative","positive","positive","positive","positive","positive","positive","positive","positive","positive","positive","positive","positive","positive","positive","positive","positive","positive","positive","positive","positive","positive","positive","negative","negative","negative","negative","negative","negative","negative","negative","negative","negative","negative","negative","negative","negative","negative","negative","negative","negative","negative","negative","positive","positive","positive","positive","positive","positive","positive","positive","positive","positive","positive","positive","positive","positive","positive","positive","positive","positive","positive","positive","positive","positive","positive"
]
var mental_illness_score = [4.333333333,4.857142857,3.904761905,4.571428571,4.285714286,4.142857143,3.380952381,3.523809524,3.904761905,4.523809524,3.952380952,4.904761905,3.761904762,3.333333333,3.238095238,3.571428571,3.476190476,3.285714286,3.047619048,3.619047619,4.333333333,3.333333333,3.285714286,2.285714286,2.523809524,3,4,3.571428571,3.380952381,3.476190476,3,2.571428571,4.047619048,2.571428571,3.238095238,2.619047619,2.095238095,3.619047619,4.285714286,3.238095238,2.80952381,3.238095238,2.857142857,3.571428571,2.904761905,2.571428571,4,2.857142857,3.19047619,3.761904762,2.761904762,2.571428571,2.619047619,2.80952381,2.761904762,2.857142857,1.571428571,3.095238095,3.619047619,3.476190476,2.380952381,3.428571429,3.047619048,2,3.19047619,2.952380952,1.714285714,1.952380952,3.952380952,2.714285714,4.142857143,2.666666667,3.714285714,2.761904762,4.714285714,3.714285714,2.666666667,3.142857143,2.904761905,3.238095238,3.761904762,2.428571429,2.761904762,3,2.619047619,3.619047619,2.285714286,3.285714286,3.619047619,4.238095238,3.952380952,4.285714286,3.19047619,2.619047619,2.428571429,3.19047619,2.571428571,2.380952381,2,2.142857143,4.095238095,3.571428571,3.904761905,3.428571429,2.952380952,2.666666667,2.285714286,2.952380952,2.80952381,2.380952381,2.666666667
]
var subjIDs = _.range(mental_illness_score.length)

// Combining the above dataset
var obsData = map(function(datum) {return _.zipObject(
  ["subjID", "mentalScore" , "riskTaking", "diseaseCondition", "frame"], datum)},
               _.zip(subjIDs, mental_illness_score, risk, disease_type, frame)); 


// The model can predict what action an agent can take given the conditions
var dataAnalysisModel = function(){
  var alpha = uniform(0, 5);
  var cognitiveModel = softMaxAgent("initial", 5, alpha);
  map(function(d){observe(cognitiveModel, d)}, obsData)
  return {
    alpha: alpha,
    french_prediction: Math.exp(cognitiveModel.score("risk"))
  }
}

var numSamples = 50000;

var inferOpts = {
  method: "MCMC",
  samples: numSamples,
  burn: numSamples / 2,
  callbacks: [editor.MCMCProgress()],
  model: dataAnalysisModel
}

var posterior = Infer(inferOpts);

viz(posterior);
~~~~
