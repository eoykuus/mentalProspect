# Belief towards mental illness as an underlying factor for risk taking in different disease scenarios
The current project captures the usage of WebPPL to conceptualize risk taking and belief towards mental illness. The model suggested here performs similarly to a human agent, and it can be used to test different scenarios.


```

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

```
