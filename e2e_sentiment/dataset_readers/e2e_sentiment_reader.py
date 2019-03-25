from typing import Dict
import json
import logging

from typing import Dict, List, Optional, Iterator
from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, SpanField, ListField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("e2e_sent")
class EntityPairwiseSentimentDatasetReader(DatasetReader):
    """
    Reads a JSON-lines file containing papers from the Semantic Scholar database, and creates a
    dataset suitable for document classification using these papers.
    Expected format for each input line: {"paperAbstract": "text", "title": "text", "venue": "text"}
    The JSON could have other fields, too, but they are ignored.
    The output of ``read`` is a list of ``Instance`` s with the fields:
        title: ``TextField``
        abstract: ``TextField``
        label: ``LabelField``
    where the ``label`` is derived from the venue of the paper.
    Parameters
    ----------
    lazy : ``bool`` (optional, default=False)
        Passed to ``DatasetReader``.  If this is ``True``, training will start sooner, but will
        take longer per batch.  This also allows training with datasets that are too large to fit
        in memory.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define input token representations. Defaults to ``{"tokens":
        SingleIdTokenIndexer()}``.
    """
    def __init__(self,
                 lazy: bool = False,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path):
        logger.info("Reading instances from lines in file at: %s", file_path)
        with open(cached_path(file_path), "r") as data_file:
            for line in data_file:
                line = line.strip("\n")
                if not line:
                    continue
                instance_json = json.loads(line)
                holder = instance_json['holder']
                target = instance_json['target']

                label = instance_json['label']
                tokens = instance_json['token']
                holder_idx = instance_json['holder_index']
                target_idx = instance_json['target_index']
                yield self.text_to_instance(tokens, holder_idx, target_idx, holder, target, label)

    @overrides
    def text_to_instance(self, tokens: List[str], holder_idx: List[List[int]], target_idx: List[List[int]],
                         holder: str, target: str, label: str = None) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        tokens_field = TextField([Token(t) for t in tokens], self._token_indexers)
        holder_field = ListField([SpanField(span[0], span[1], tokens_field) for span in holder_idx])
        target_field = ListField([SpanField(span[0], span[1], tokens_field) for span in target_idx])

        metadata = {'token': tokens, 'holder_index': holder_idx, 'target_index': target_idx,
                    'holder': holder, 'target': target}
        fields = {'tokens': tokens_field, 'holder': holder_field, 'target': target_field}
        if label is not None:
            metadata['label'] = label
            fields['label'] = LabelField(label)
        metadata_field = MetadataField(metadata)
        fields['metadata'] = metadata_field

        return Instance(fields)
