# Task 1: MRP (minimum required project)

## The task

Goal: Segment images of dogs and cats

Dataset: Oxford IIIT pet dataset

Scope: Should we evaluate outside the dataset too? ie. "real-world performance"
The dataset only has 7349 total images - which is barely enough to train.

Task: Simple semantic segmentation (not instance or panoptic segmentation). In
particular, we want to segment the pet from the rest of the image, we do not
care about segmenting any other objects. (binary segmentation)

The model is trained for inputs containing exactly 1 front-facing pet. Inputs
that do not contain a pet are disregarded - we do not evaluate these cases. 
Inputs containing more than 1 pet (2-3) may be evaluated as a side measurement,
but not as part of the main evaluation.

## Limitations

The dataset consists of single pet images (dogs and cats), hence when the model
is evaluated on "real world data", it may be susceptible to:

- Multiple front-facing pets in the image (training data has only 1 ff pet)
- Different animals (in the same picture as the dog/cat pet)
- Different breeds of dogs and cats (color, shape, size, etc.)
- Images of pets not front-facing (cannot see face)



# Task 2: OEQ (open ended question)

TBD based on model architecture performance.
