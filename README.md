---

## Installation

The library is installable in any python script/notebook as
```
git clone https://github.com/kartikparnami/explainable_ai.git
```

We plan to support PyPi-based pip installation in the future.

## Usage

To explain any instance, first import the explainer from the library as follows:

```
from explainable_ai.(anchor/counterfactual/integrated_gradients).<method_name>_(image/text) import <class_name>

# Example:
from explainable_ai.anchor.anchor_image import AnchorImage
from explainable_ai.counterfactual.lc_text import LimeCounterfactualText
```

Initialize the explanation instance (works fine with default parameters in most cases):

```
explainer = XXXExplainer()

# Example:
explainer = AnchorImage()
explainer = LimeCounterfactualText()
```

Now use the explainer instance to explain predictions:
```
explanation = explainer.explain_instance(instance, model/predict_fn, ...)

# Example:
explanation = anchor_image_explainer.explain(image, predict_fn)
plt.imshow(explanation)

explanation = lime_counterfactual_text_explainer.explain(text, predict_fn)
explanation
```

Note: Integrated gradients technique requires to pass the TensorFlow model itself as it is a whitebos technique which works by accessing the model weights.

## Examples

Examples of usage of each of these methods can be found in examples/ folder in jupyter notebooks.

#### Sample image explanations:

##### Anchor Image Explanation
![Anchor Image Explanation](images/AnchorImage.png "Anchor Image Explanation")

##### Counterfactual Image Explanation
![Counterfactual Image Explanation](images/LCImage.png "Counterfactual Image Explanation")

##### Integrated Gradients Image Explanation
![Integrated Gradients Image Explanation](images/IGImage.png "Integrated Gradients Image Explanation")

#### Sample text explanations:

##### Anchor Text Explanation
![Anchor Text Explanation](images/AnchorText.png "Anchor Text Explanation")

##### Counterfactual Text Explanation
![Counterfactual Text Explanation](images/LCText.png "Counterfactual Text Explanation")

##### Integrated Gradients Text Explanation
![Integrated Gradients Text Explanation](images/IGText.png "Integrated Gradients Text Explanation")
