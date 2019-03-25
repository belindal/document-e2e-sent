
from typing import Dict, Optional

import numpy
from overrides import overrides
import torch
import torch.nn.functional as F
from torch import nn

from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import FeedForward, Seq2SeqEncoder, TextFieldEmbedder
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy, F1Measure
from allennlp.modules.span_extractors import SelfAttentiveSpanExtractor, EndpointSpanExtractor

import pdb


@Model.register("rnn_classifier")
class RNNClassifier(Model):
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 classifier_feedforward: FeedForward,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(RNNClassifier, self).__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.num_classes = self.vocab.get_vocab_size("labels")
        self.encoder = encoder
        self.endpoint_span_extractor = EndpointSpanExtractor(encoder.get_output_dim(),
                                                             combination="x,y")
        self.attentive_span_extractor = SelfAttentiveSpanExtractor(encoder.get_output_dim())

        attention_input_dim = encoder.get_output_dim() * 2
        self.holder_attention = nn.Linear(attention_input_dim, 1)
        self.target_attention = nn.Linear(attention_input_dim, 1)

        self.classifier_feedforward = classifier_feedforward

        if text_field_embedder.get_output_dim() != encoder.get_input_dim():
            raise ConfigurationError("The output dimension of the text_field_embedder must match the "
                                     "input dimension of the title_encoder. Found {} and {}, "
                                     "respectively.".format(text_field_embedder.get_output_dim(),
                                                            encoder.get_input_dim()))
        self.metrics = {
                "f1_neg": F1Measure(1),
                "f1_none": F1Measure(0),
                "f1_pos": F1Measure(2),
        }
        self.loss = torch.nn.CrossEntropyLoss()

        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],
                holder: torch.IntTensor,
                target: torch.IntTensor,
                metadata: Dict[str, str],
                label: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        title : Dict[str, Variable], required
            The output of ``TextField.as_array()``.
        abstract : Dict[str, Variable], required
            The output of ``TextField.as_array()``.
        label : Variable, optional (default = None)
            A variable representing the label for each instance in the batch.
        Returns
        -------
        An output dictionary consisting of:
        class_probabilities : torch.FloatTensor
            A tensor of shape ``(batch_size, num_classes)`` representing a distribution over the
            label classes for each instance.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """
        embedded_tokens = self.text_field_embedder(tokens)
        tokens_mask = util.get_text_field_mask(tokens)
        encoded_tokens = self.encoder(embedded_tokens, tokens_mask)

        holder_span_mask = (holder[:, :, 0] >= 0).squeeze(-1).long()
        target_span_mask = (target[:, :, 0] >= 0).squeeze(-1).long()
        holder_endpoint_span_embeddings = self.endpoint_span_extractor(encoded_tokens, holder, tokens_mask,
                                                                       holder_span_mask)
        target_endpoint_span_embeddings = self.endpoint_span_extractor(encoded_tokens, target, tokens_mask,
                                                                       target_span_mask)
        # holder_attentive_span_embeddings = self.attentive_span_extractor(encoded_tokens, holder, tokens_mask,
        #                                                                  holder_span_mask)
        # target_attentive_span_embeddings = self.attentive_span_extractor(encoded_tokens, target, tokens_mask,
        #                                                                  target_span_mask)
        holder_span_attention_logits = self.holder_attention(holder_endpoint_span_embeddings).squeeze()
        holder_span_attention_weights = util.masked_softmax(holder_span_attention_logits, holder_span_mask)
        attended_holder_embeddings = util.weighted_sum(holder_endpoint_span_embeddings, holder_span_attention_weights)

        target_span_attention_logits = self.target_attention(target_endpoint_span_embeddings).squeeze()
        target_span_attention_weights = util.masked_softmax(target_span_attention_logits, target_span_mask)
        attended_target_embeddings = util.weighted_sum(target_endpoint_span_embeddings, target_span_attention_weights)

        logits = self.classifier_feedforward(
            torch.cat([attended_holder_embeddings, attended_target_embeddings], dim=-1)
        )
        output_dict = {'logits': logits}
        if label is not None:
            loss = self.loss(logits, label)
            for metric in self.metrics.values():
                metric(logits, label)
            output_dict["loss"] = loss

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does a simple argmax over the class probabilities, converts indices to string labels, and
        adds a ``"label"`` key to the dictionary with the result.
        """
        class_probabilities = F.softmax(output_dict['logits'], dim=-1)
        output_dict['class_probabilities'] = class_probabilities

        predictions = class_probabilities.cpu().data.numpy()
        argmax_indices = numpy.argmax(predictions, axis=-1)
        labels = [self.vocab.get_token_from_index(x, namespace="labels")
                  for x in argmax_indices]
        output_dict['label'] = labels
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metric_dict = {}
        f1_scores = []
        for metric_name in self.metrics:
            precision, recall, f1 = self.metrics[metric_name].get_metric(reset)
            metric_dict[metric_name] = f1
            f1_scores.append(f1)
        metric_dict["f1"] = float(sum(f1_scores)) / len(f1_scores)
        return metric_dict
