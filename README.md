# TableGraph
This is the prototype model and dataset of the TableGraph work. 
TableGraph is designed to automatically interpret and understand table data with parsing table images to table cell graphs.
We believe graph would be the best representation style for all kind of table data.
(Including Common Situation with Uniform Row and Column Indexes or Nested Situation with Uneven Row and Column Indexes)

The key contribution of TableGraph is transforming table data interpretation to be a semantic segmentation issue. 
With that the complexity of table structure variation can be eliminated. 
Since no matter how complex of the table structure (especially for the nested situation), there would be three semantic classes (borderline, content, background).

Also, a comprehensive dataset is crucial to train such a semantic segmentation model. 
In addition of employing DeepLabv3+ as the backbone model to construct a robust interpretation model.
A table image generation algorithm is designed, thus theoretically we can obtain unlimited training data.
What interesting is that with doing so we do not need to annotate these table image data, since the semantic annotation mask can be simultaneously generated.

Here we publish our first try, now it includes three core parts:
(1) A table image and corresponding annotation mask image generation algorithm.
(2) A DeepLabv3+ based table image semantic segmentation model.
(3) A table image dataset.

We have to admit that this is very intuitive work. 
Although we have established some tests and obtained good results, there are limitations, and we are going to conduct further investigation.
We hope it can be a systematic work in the near future.
