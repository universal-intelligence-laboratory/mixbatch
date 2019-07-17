# mixbatch
The mixbatch act as a interpolate to feature space, to make a soft decition margain to enhance generalization.
And the good part is, you can apply mixbatch on any layer you like, any.

## TODO
auto-adjust alpha of mixup layer:
- with train noise of layer?
- with grad?
- with linear schedule of epoch?

NAS run for best insert layer(s):
- call nni
- call nni with multi layer selected