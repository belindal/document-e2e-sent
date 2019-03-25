{
  "dataset_reader": {
    "type": "e2e_sent"
  },
  "train_data_path": "./data/new_annot/new_train.json",
  "validation_data_path": "./data/new_annot/acl_dev_tune_new.json",
  "model": {
    "type": "rnn_classifier",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          //"pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.300d.txt.gz",
          "embedding_dim": 300,
          //"trainable": false
        }
      }
    },
    "encoder": {
      "type": "lstm",
      "bidirectional": true,
      "input_size": 300,
      "hidden_size": 100,
      "num_layers": 1,
      "dropout": 0.2
    },
    "classifier_feedforward": {
      "input_dim": 800,
      "num_layers": 2,
      "hidden_dims": [400, 3],
      "activations": ["relu", "linear"],
      "dropout": [0.2, 0.0]
    }
  },
  "iterator": {
    "type": "bucket",
    "sorting_keys": [["tokens", "num_tokens"]],
    "batch_size": 32
  },
  "trainer": {
    "num_epochs": 40,
    "patience": 10,
    "cuda_device": 0,
    "grad_clipping": 5.0,
    "validation_metric": "+f1",
    "optimizer": {
      "type": "adagrad"
    }
  }
}