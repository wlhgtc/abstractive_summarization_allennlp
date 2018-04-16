from typing import Tuple,List
import simplejson as json
from overrides import overrides

from allennlp.common.util import JsonDict,sanitize
from allennlp.data import Instance
from allennlp.service.predictors.predictor import Predictor

@Predictor.register('abstract-generator')
class AbstractGeneratorPredictor(Predictor):
    """"Predictor wrapper for the AbstractGeneratorPredictor"""
    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Tuple[Instance, JsonDict]:
        index = json_dict.get('index',None)
        article = json_dict['article']
        instance = self._dataset_reader.text_to_instance(source_string=article)
        return instance,{'index':index} if index is not None else {} 
    
    @overrides
    def predict_json(self, inputs: JsonDict, cuda_device: int = -1) -> JsonDict:
        instance, return_dict = self._json_to_instance(inputs)
        outputs = self._model.forward_on_instance(instance, cuda_device)
        return_dict.update({'summarization':''.join(outputs['predicted_tokens'])})
        return sanitize(return_dict)
    
    @overrides
    def predict_batch_json(self, inputs: List[JsonDict], cuda_device: int = -1) -> List[JsonDict]:
        instances, return_dicts = zip(*self._batch_json_to_instances(inputs))
        outputs = self._model.forward_on_instances(instances, cuda_device)
        for output, return_dict in zip(outputs, return_dicts):
            return_dict.update({'summarization':''.join(output['predicted_tokens'])})
        return sanitize(return_dicts)
   
    @overrides
    def dump_line(self, outputs: JsonDict) -> str:  # pylint: disable=no-self-use
        """
        If you don't want your outputs in JSON-lines format
        you can override this function to output them differently.
        """
        return json.dumps(outputs,ensure_ascii=False,encoding="utf-8") + "\n"