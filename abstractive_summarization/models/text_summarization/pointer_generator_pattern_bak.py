from typing import Dict, List, Union
from collections import defaultdict
import numpy
import numba as nb
from overrides import overrides

import torch
from torch.nn.modules.rnn import LSTMCell
from torch.nn.modules.linear import Linear
import torch.nn.functional as F

from allennlp.common import Params
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.dataset import Batch
from allennlp.data.dataset_readers.seq2seq import START_SYMBOL, END_SYMBOL
from allennlp.modules import Attention, TextFieldEmbedder, Seq2SeqEncoder
from allennlp.modules.similarity_functions import SimilarityFunction
from allennlp.modules.token_embedders import Embedding
from allennlp.models.model import Model
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits, weighted_sum, get_final_encoder_states,masked_softmax
from allennlp.data.instance import Instance

from abstractive_summarization.training.metrics.rouge import Rouge


@Model.register("pgp")
class PointerGeneratorPattern(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 source_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 max_decoding_steps: int,
                 target_namespace: str = "tokens",
                 target_embedder: TextFieldEmbedder = None,
                 attention_function: SimilarityFunction = None,
                 scheduled_sampling_ratio: float = 0.25
                 ) -> None:
        super(PointerGeneratorPattern, self).__init__(vocab)
        self._source_embedder = source_embedder
        self._encoder = encoder
        self._max_decoding_steps = max_decoding_steps
        self._target_namespace = target_namespace
        self._attention_function = attention_function
        self._scheduled_sampling_ratio = scheduled_sampling_ratio
        self._pattern_pos = ['@@np@@','@@ns@@','@@ni@@','@@nz@@','@@m@@','@@i@@','@@id@@','@@t@@','@@j@@']
        self._start_index = self.vocab.get_token_index(START_SYMBOL, self._target_namespace)
        self._end_index = self.vocab.get_token_index(END_SYMBOL, self._target_namespace)
        num_classes = self.vocab.get_vocab_size(self._target_namespace)

        self._target_embedder = target_embedder or source_embedder
        #!!! attention on decoder output, not on decoder input !!!#
        self._decoder_input_dim = self._target_embedder.get_output_dim()
                
        # decoder use UniLSTM while encoder use BiLSTM 
        self._decoder_hidden_dim = self._encoder.get_output_dim()
        
        # decoder: h0 c0 projection_layer from final_encoder_output
        self.decode_h0_projection_layer = Linear(self._encoder.get_output_dim(),self._decoder_hidden_dim)
        self.decode_c0_projection_layer = Linear(self._encoder.get_output_dim(),self._decoder_hidden_dim)

        self._decoder_attention = Attention(self._attention_function)
        # The output of attention, a weighted average over encoder outputs, will be
        # concatenated to the decoder_hidden of the decoder at each time step.
        # V[s_t, h*_t] + b
        self._decoder_output_dim = self._decoder_hidden_dim + self._encoder.get_output_dim() #[s_t, h*_t]
        
        # TODO (pradeep): Do not hardcode decoder cell type.
        self._decoder_cell = LSTMCell(self._decoder_input_dim, self._decoder_hidden_dim)
        self._output_attention_layer = Linear(self._decoder_output_dim, self._decoder_hidden_dim)
        #V[s_t, h*_t] + b
        self._output_projection_layer = Linear(self._decoder_hidden_dim, num_classes)
        # num_classes->V'
        # generationp robability
        self._pointer_gen_layer = Linear(self._decoder_hidden_dim+self._encoder.get_output_dim()+self._decoder_input_dim, 1)
        # metrics
        self.metrics = {
                "ROUGE-1": Rouge(1),
                "ROUGE-2": Rouge(2),
        }

    @overrides
    def forward(self,  # type: ignore
                source_tokens: Dict[str, torch.LongTensor] = None,
                target_tokens: Dict[str, torch.LongTensor] = None,
                source_tokens_raw = None,
                target_tokens_raw = None,
                predict: bool = False) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        embedded_input = self._source_embedder(source_tokens)
        batch_size, _, _ = embedded_input.size()
        source_mask = get_text_field_mask(source_tokens)
        encoder_outputs = self._encoder(embedded_input, source_mask)
        final_encoder_output = get_final_encoder_states(encoder_outputs,source_mask)#encoder_outputs[:, -1]  # (batch_size, encoder_output_dim)
        if target_tokens:
            targets = target_tokens["tokens"]
            target_sequence_length = targets.size()[1]
            # The last input from the target is either padding or the end symbol. Either way, we
            # don't have to process it.
            num_decoding_steps = target_sequence_length - 1
        else:
            num_decoding_steps = self._max_decoding_steps
        decoder_hidden = self.decode_h0_projection_layer(final_encoder_output)
        decoder_context = self.decode_h0_projection_layer(final_encoder_output)
        last_predictions = None
        step_attensions = []
        step_probabilities = []
        step_predictions = []
        step_p_gen = []
        for timestep in range(num_decoding_steps):
            if self.training and all(torch.rand(1) >= self._scheduled_sampling_ratio):
                input_choices = targets[:, timestep]
            else:
                if timestep == 0:
                    # For the first timestep, when we do not have targets, we input start symbols.
                    # (batch_size,)
                    input_choices = source_mask.new().resize_(batch_size).fill_(self._start_index)
                else:
                    input_choices = last_predictions
            # input_indices : (batch_size,)  since we are processing these one timestep at a time.
            # (batch_size, target_embedding_dim)
            input_choices = {'tokens':input_choices}
            decoder_input = self._target_embedder(input_choices)
            #Dh_t(S_t),Dc_t
            decoder_hidden, decoder_context = self._decoder_cell(decoder_input,
                                                                 (decoder_hidden, decoder_context))
                                                                 
            #cat[S_t,H*_t(short memory)] 
            P_attensions, decoder_output = self._decode_step_output(decoder_hidden, encoder_outputs, source_mask)
            # (batch_size, num_classes)
            # W[S_t,H*_t]+b
            output_attention = self._output_attention_layer(decoder_output) 
            output_projections = self._output_projection_layer(output_attention)
            # P_vocab
            class_probabilities = F.softmax(output_projections, dim=-1)
            # generation probability
            #P_gen = F.sigmoid(self._pointer_gen_layer(torch.cat((decoder_output,decoder_input),-1)))
            #class_probabilities = P_gen*class_probabilities
            #step_p_gen.append(P_gen.unsqueeze(1))
            #print(f'P_gen:{P_gen.data.mean()}')
            # list of (batch_size, 1, num_classes)
            step_attensions.append(P_attensions.unsqueeze(1))
            _, predicted_classes = torch.max(class_probabilities, 1)
            step_probabilities.append(class_probabilities.unsqueeze(1))
            last_predictions = predicted_classes
            # (batch_size, 1)
            step_predictions.append(last_predictions.unsqueeze(1))
        # This is (batch_size, num_decoding_steps, num_classes)
        all_attensions = torch.cat(step_attensions, 1)
        #all_p_gens = torch.cat(step_p_gen,1)
        class_probabilities = torch.cat(step_probabilities, 1)
        all_predictions = torch.cat(step_predictions, 1)
        output_dict = {"all_attensions": all_attensions,
                       #"all_p_gens": all_p_gens,
                       "source_tokens": source_tokens_raw,
                       "class_probabilities": class_probabilities,
                       "predictions": all_predictions}
        #att_dists = self._att_dists(all_predictions,all_attensions,source_tokens_raw)
        #output_dict.update({"att_dists":att_dists})
        if target_tokens:
            target_mask = get_text_field_mask(target_tokens)
            gen_loss = self._get_loss(class_probabilities, targets, target_mask)
            import pdb
            pdb.set_trace()
            #copy_loss = self._get_copy_loss(all_p_gens,att_dists,target_tokens_raw)
            #copy_loss = self._get_copy_loss(att_dists,target_tokens_raw)
            #loss = gen_loss#+copy_loss
            print(f'gen_loss:{gen_loss.data.mean()}')#,copy_loss:{copy_loss.data.mean()}')
            output_dict["loss"] = gen_loss
            for metric in self.metrics.values():
                evaluated_sentences = [''.join(i) for i in self.decode(output_dict)["predicted_tokens"]]
                reference_sentences = [''.join([j.text for j in i]) for i in target_tokens_raw]
                #print(f'evaluated_sentences:{evaluated_sentences},reference_sentences:{reference_sentences}')
                metric(evaluated_sentences,reference_sentences)

        return output_dict

    def _decode_step_output(self,
                            decoder_hidden_state: torch.LongTensor = None,
                            encoder_outputs: torch.LongTensor = None,
                            encoder_outputs_mask: torch.LongTensor = None) -> torch.LongTensor:
        encoder_outputs_mask = encoder_outputs_mask.float()
        input_weights_e = self._decoder_attention(decoder_hidden_state, encoder_outputs, encoder_outputs_mask)
        input_weights_a = masked_softmax(input_weights_e,encoder_outputs_mask)
        attended_input = weighted_sum(encoder_outputs, input_weights_a)
        return input_weights_a,torch.cat((decoder_hidden_state,attended_input), -1)

    @staticmethod
    def _get_loss(probs: torch.LongTensor,
                  targets: torch.LongTensor,
                  target_mask: torch.LongTensor,
                  batch_average: bool = True) -> torch.LongTensor:
        relevant_targets = targets[:, 1:].contiguous()  # (batch_size, num_decoding_steps)
        relevant_mask = target_mask[:, 1:].contiguous()  # (batch_size, num_decoding_steps)
        probs_flat = probs.view(-1,probs.size(-1))
        log_probs_flat = torch.log(probs_flat)
        targets_flat = relevant_targets.view(-1, 1).long()
        
        negative_log_likelihood_flat = - torch.gather(log_probs_flat, dim=1, index=targets_flat)
        # shape : (batch, sequence_length)
        negative_log_likelihood = negative_log_likelihood_flat.view(*relevant_targets.size())
        # shape : (batch, sequence_length)
        negative_log_likelihood = negative_log_likelihood * relevant_mask.float()
        # shape : (batch_size,)
        per_batch_loss = negative_log_likelihood.sum(1) / (relevant_mask.sum(1).float() + 1e-13)

        if batch_average:
            num_non_empty_sequences = ((relevant_mask.sum(1) > 0).float().sum() + 1e-13)
            return per_batch_loss.sum() / num_non_empty_sequences
        return per_batch_loss

    #@nb.jit
    def _get_copy_loss(self,att_dists,targets):
        copy_loss = torch.zeros(1).fill_(1e-13).cuda()
        #copy_loss = copy_loss.cuda() if att_dists[0][0].is_cuda else copy_loss
        for att_dist,target in zip(att_dists,targets):
            for step_att_dist,token in zip(att_dist,target):
                if '@@'+token.pos_+'@@' in self._pattern_pos and token.text in step_att_dist.keys():
                    print(f'pattern:{token}')
                    copy_loss += -torch.log(step_att_dist[token.text])#*(1-p_gen))
                # elif p_gen.item() < 0.5 and token.text in step_att_dist.keys():
                    # print(f'word:{token}')
                    # copy_loss += -torch.log(step_att_dist[token.text]*(1-p_gen))
                else:
                    continue
        return copy_loss
        
    @nb.jit
    def _att_dists(self,predictions,attensions,sources):
        att_dist = ['_']
        if not isinstance(predictions, numpy.ndarray):
            predictions = predictions.data.cpu().numpy()
        for prediction,attension,source in zip(predictions,attensions,sources):
            att_dist_temp = []
            indices = list(prediction)
            if self._end_index in indices:
                attension = attension[:indices.index(self._end_index)]
            for step_attension in attension:
                att_dist_temp.append(self._att_dist(source,step_attension))
            att_dist.append(att_dist_temp)
        return att_dist[1:]
        
    @staticmethod
    @nb.jit
    def _att_dist(source,attension):
        att_dist = defaultdict()
        attension = attension[1:len(source)+1]
        for token,att in zip(source,attension):
            if token in att_dist.keys():
                att_dist[token.text] += att
            else:
                att_dist[token.text] = att
        return att_dist

    @overrides
    #@nb.jit
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        predicted_indices = output_dict["predictions"]
        #all_p_gens = output_dict["all_p_gens"]
        #att_dists = output_dict["att_dists"]
        if not isinstance(predicted_indices, numpy.ndarray):
            predicted_indices = predicted_indices.data.cpu().numpy()
        all_predicted_tokens = []
        for indices in predicted_indices:
            indices = list(indices)
            # Collect indices till the first end_symbol
            if self._end_index in indices:
                end_index = indices.index(self._end_index)
                indices = indices[:end_index]
                #p_gens = p_gens[:end_index]
                #att_dist = att_dist[:end_index]
            predicted_tokens = [self.vocab.get_token_from_index(x, namespace=self._target_namespace)
                                for x in indices]
            # predicted_tokens_ = []
            # for token,step_att_dist in zip(predicted_tokens,att_dist):
                # import pdb
                # pdb.set_trace()
                # if token in self._pattern_pos:
                    # predicted_tokens_.append(sorted(step_att_dist.items(),key=lambda x:x[1].item(),reverse=True)[0][0])
                # else:
                    # predicted_tokens_.append(token)
                # elif p_gen.item() > 0.5:
                    # predicted_tokens_.append(token)
                # else:
                    # predicted_tokens_.append(sorted(step_att_dist.items(),key=lambda x:x[1].item(),reverse=True)[0][0])
                    
            all_predicted_tokens.append(predicted_tokens)
        output_dict["predicted_tokens"] = all_predicted_tokens
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}

    @overrides
    def forward_on_instances(self,
                             instances: List[Instance],
                             cuda_device: int) -> List[Dict[str, numpy.ndarray]]:
        model_input = {}
        dataset = Batch(instances)
        dataset.index_instances(self.vocab)
        model_input = dataset.as_tensor_dict(cuda_device=cuda_device, for_training=False)
        #input
        model_input.update({'instances':instances})
        model_input.update({'predict':True})
        outputs = self.decode(self(**model_input))

        instance_separated_output: List[Dict[str, numpy.ndarray]] = [{} for _ in dataset.instances]
        for name, output in list(outputs.items()):
            if output.is_cuda:
                output = output.cpu().numpy()
            outputs[name] = output
            for instance_output, batch_element in zip(instance_separated_output, output):
                instance_output[name] = batch_element
        return instance_separated_output

    @classmethod
    def from_params(cls, vocab, params: Params) -> 'PointerGeneratorPattern':
        source_embedder_params = params.pop("source_embedder")
        source_embedder = TextFieldEmbedder.from_params(vocab, source_embedder_params)
        encoder = Seq2SeqEncoder.from_params(params.pop("encoder"))
        max_decoding_steps = params.pop("max_decoding_steps")
        target_namespace = params.pop("target_namespace", "tokens")
        # If no attention function is specified, we should not use attention, not attention with
        # default similarity function.
        attention_function_type = params.pop("attention_function", None)
        if attention_function_type is not None:
            attention_function = SimilarityFunction.from_params(attention_function_type)
        else:
            attention_function = None
        scheduled_sampling_ratio = params.pop_float("scheduled_sampling_ratio", 0.25)
        params.assert_empty(cls.__name__)
        return cls(vocab,
                   source_embedder=source_embedder,
                   encoder=encoder,
                   max_decoding_steps=max_decoding_steps,
                   target_namespace=target_namespace,
                   attention_function=attention_function,
                   scheduled_sampling_ratio=scheduled_sampling_ratio
                   )
