# fair-xg

````Python
from fairness.fairness import FairnessEvaluator

fe = FairnessEvaluator(df=df, target='target', preds='shot_statsbomb_xg')
fe.fit(sensitives=['gender'])
````
