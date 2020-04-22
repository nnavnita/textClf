# textClf

textClf is a Python text classifier built using [Flair](https://github.com/zalandoresearch/flair).

## Dependencies

To install the dependencies:

```
pip install allennlp flair==0.4.3 nltk pandas sklearn tqdm
```

## Usage

To train the model:
```
python main.py train [file].csv [param].json
```

To use the model to make predictions:
```
python main.py predict [file].csv [param].json
```

## Next Steps

- Multi-task learning to learn multiple labels (multi-label multi-class classification)
- Implement retraining (existing model can be retrained with new data)
