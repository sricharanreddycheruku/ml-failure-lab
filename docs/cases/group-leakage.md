# Repeated entities in train and test

## Question

Can the reported test score tell us how a classifier behaves on entities absent from training?

An entity is a person, device, patient or other unit with multiple records. Training data is the data used to fit a model. Test data is held back to evaluate it. Accuracy is the fraction of test records for which the predicted label matches the true label.

## The fixture

The case creates 40 fictional entities, each with two identical records. The label is zero or one. The feature is a numeric fingerprint for the entity. It gives the classifier a way to recognize that entity, but no designed signal for predicting a new entity's label.

The first 30 entities and last 10 entities each have balanced labels. Labels are shuffled with a fixed seed. This fixture is deliberately small and artificial.

## Incorrect for the declared task

Put the first record of every entity in training and its second record in testing. The rows are separate, but every test entity is already known.

A one-nearest-neighbor classifier predicts the label of the training record with the closest feature. Every test fingerprint has an exact match. The measured accuracy is 100% with 40 shared entity IDs.

That can be a valid measurement of recognizing more records of known entities. It does not answer the stated question about previously unseen entities.

## Correction

Put both records of each of the first 30 entities in training and both records of each of the remaining 10 entities in testing. There are no shared entity IDs.

In this fixture, the classifier always chooses the nearest training fingerprint for the new entities and gets 50% accuracy. The correction changes the evaluation population. The exact percentages and the different record counts belong to this fixture.

## Test the requirement

```text
python -m ml_failure_lab run group-leakage
python -m ml_failure_lab verify group-leakage --approach bad
python -m ml_failure_lab verify group-leakage --approach fixed
```

The check computes the intersection of the training and test entity-ID sets. An intersection is the set of IDs present in both. For an evaluation of unseen entities, that intersection must be empty.

The `bad` verification exits with 1. The `fixed` verification exits with 0. The tests confirm both outcomes.

## Limits and reference

Real grouped data may have overlapping, correlated or hierarchical identities. The right split depends on what the deployed system must predict. This case does not establish the size of leakage effects on another dataset.

See [scikit-learn's guidance on grouped cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data). The fixture and classifier in this case were written for this project.
