from typing import Dict, List, Union

import numpy
from overrides import overrides

import torch
from torch.autograd import Variable
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
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits, weighted_sum
from allennlp.data.instance import Instance


@Model.register("mt")
class MultiTask(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 source_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 max_decoding_steps: int,
                 target_namespace: str = "tokens",
                 target_embedder: TextFieldEmbedder = None,
                 attention_function: SimilarityFunction = None,
                 scheduled_sampling_ratio: float = 0.25,
                 pointer_gen: bool = True,
                 max_oovs: int = None) -> None:
        super(MultiTask, self).__init__(vocab)
        self._source_embedder = source_embedder
        self._encoder = encoder
        self._max_decoding_steps = max_decoding_steps
        self._target_namespace = target_namespace
        self._attention_function = attention_function
        self._scheduled_sampling_ratio = scheduled_sampling_ratio
        self._pointer_gen = pointer_gen
        if self._pointer_gen:
            self._max_oovs = max_oovs
            self.vocab.set_max_oovs(self._max_oovs)
        # We need the start symbol to provide as the input at the first timestep of decoding, and
        # end symbol as a way to indicate the end of the decoded sequence.
        self._start_index = self.vocab.get_token_index(START_SYMBOL, self._target_namespace)
        self._end_index = self.vocab.get_token_index(END_SYMBOL, self._target_namespace)
        num_classes = self.vocab.get_vocab_size(self._target_namespace)
        # Decoder output dim needs to be the same as the encoder output dim since we initialize the
        # hidden state of the decoder with that of the final hidden states of the encoder. Also, if
        # we're using attention with ``DotProductSimilarity``, this is needed.

        self._target_embedder = target_embedder or source_embedder
        #!!! attention on decoder output, not on decoder input !!!#
        self._decoder_input_dim = self._target_embedder.get_output_dim()
                
        # decoder use UniLSTM while encoder use BiLSTM 
        self._decoder_hidden_dim = self._encoder.get_output_dim()//2
        
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
        if self._pointer_gen:
            self._pointer_gen_layer = Linear(self._decoder_hidden_dim+self._encoder.get_output_dim()+self._decoder_input_dim, 1)

    @overrides
    def forward(self,  # type: ignore
                source_tokens: Dict[str, torch.LongTensor] = None,
                target_tokens: Dict[str, torch.LongTensor] = None,
                raw: Dict = None,
                extended: Dict = None,
                predict: bool = False) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Decoder logic for producing the entire target sequence.

        Parameters
        ----------
        source_tokens : Dict[str, torch.LongTensor]
           The output of ``TextField.as_array()`` applied on the source ``TextField``. This will be
           passed through a ``TextFieldEmbedder`` and then through an encoder.
        target_tokens : Dict[str, torch.LongTensor], optional (default = None)
           Output of ``Textfield.as_array()`` applied on target ``TextField``. We assume that the
           target tokens are also represented as a ``TextField``.
        """
        if self._pointer_gen:
            source_tokens = raw['source_tokens']
            if 'target_tokens' in raw.keys():
                target_tokens = raw['target_tokens']
        embedded_input = self._source_embedder(source_tokens)
        batch_size, _, _ = embedded_input.size()
        source_mask = get_text_field_mask(source_tokens)
        encoder_outputs = self._encoder(embedded_input, source_mask)
        final_encoder_output = encoder_outputs[:, -1]  # (batch_size, encoder_output_dim)
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
        step_logits = []
        step_probabilities = []
        step_predictions = []
        for timestep in range(num_decoding_steps):
            if self.training and all(torch.rand(1) >= self._scheduled_sampling_ratio):
                input_choices = targets[:, timestep]
            else:
                if timestep == 0:
                    # For the first timestep, when we do not have targets, we input start symbols.
                    # (batch_size,)
                    input_choices = Variable(source_mask.data.new()
                                             .resize_(batch_size).fill_(self._start_index))
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
            P_attensions_e,P_attensions, decoder_output = self._decode_step_output(decoder_hidden, encoder_outputs, source_mask)
            # (batch_size, num_classes)
            # W[S_t,H*_t]+b
            output_attention = self._output_attention_layer(decoder_output) 
            output_projections = self._output_projection_layer(output_attention)
            # P_vocab
            class_probabilities = F.softmax(output_projections, dim=-1)
            if self._pointer_gen:
                # generation probability
                P_gen = F.sigmoid(self._pointer_gen_layer(torch.cat((decoder_output,decoder_input),-1)))
                P_copy = 1 - P_gen
                expand_shape =[batch_size,len(self.vocab._token_to_index['tokens'])-self.vocab.unextend_len]
                expand_variable = Variable(torch.zeros(expand_shape).cuda() if class_probabilities.is_cuda else torch.zeros(expand_shape))
                
                expand_logits = torch.cat([output_projections,expand_variable],dim=-1)
                final_logits = expand_logits*P_gen + torch.zeros_like(expand_logits).scatter_(-1,extended['source_tokens']['tokens'], P_attensions_e*P_copy)
                expand_probabilities = torch.cat([class_probabilities,expand_variable],dim=-1)
                final_probabilities = expand_probabilities*P_gen + torch.zeros_like(expand_probabilities).scatter_(-1,extended['source_tokens']['tokens'], P_attensions*P_copy)
                class_probabilities = final_probabilities
            # list of (batch_size, 1, num_classes)
            step_logits.append(final_logits.unsqueeze(1) if self._pointer_gen else output_projections.unsqueeze(1))
            _, predicted_classes = torch.max(class_probabilities, 1)
            step_probabilities.append(class_probabilities.unsqueeze(1))
            last_predictions = predicted_classes
            # (batch_size, 1)
            step_predictions.append(last_predictions.unsqueeze(1))
        # step_logits is a list containing tensors of shape (batch_size, 1, num_classes)
        # This is (batch_size, num_decoding_steps, num_classes)
        logits = torch.cat(step_logits, 1)
        class_probabilities = torch.cat(step_probabilities, 1)
        all_predictions = torch.cat(step_predictions, 1)
        output_dict = {"logits": logits,
                       "class_probabilities": class_probabilities,
                       "predictions": all_predictions}
        if target_tokens:
            target_mask = get_text_field_mask(target_tokens)
            targets = extended['target_tokens']['tokens'] if self._pointer_gen else targets
            loss = self._get_loss(logits, targets, target_mask)
            output_dict["loss"] = loss
            # TODO: Define metrics
        if not predict:
            if self._pointer_gen:
                self.vocab.drop_extend()
        return output_dict

    def _decode_step_output(self,
                            decoder_hidden_state: torch.LongTensor = None,
                            encoder_outputs: torch.LongTensor = None,
                            encoder_outputs_mask: torch.LongTensor = None) -> torch.LongTensor:
        """
        Given all the encoder outputs, compute the input at the current timestep.  Note: This method is agnostic to
        whether the indices are gold indices or the predictions made by the decoder at the last
        timestep. So, this can be used even if we're doing some kind of scheduled sampling.

        Parameters
        ----------
        decoder_hidden_state : torch.LongTensor, optional (not needed if no attention)
            Output of from the decoder at the last time step. Needed only if using attention.
        encoder_outputs : torch.LongTensor, optional (not needed if no attention)
            Encoder outputs from all time steps. Needed only if using attention.
        encoder_outputs_mask : torch.LongTensor, optional (not needed if no attention)
            Masks on encoder outputs. Needed only if using attention.
        """
        # encoder_outputs : (batch_size, input_sequence_length, encoder_output_dim)
        # Ensuring mask is also a FloatTensor. Or else the multiplication within attention will
        # complain.
        #import pdb
        #pdb.set_trace()
        encoder_outputs_mask = encoder_outputs_mask.float()
        # (batch_size, input_sequence_length)
        input_weights_e = self._decoder_attention(decoder_hidden_state, encoder_outputs, encoder_outputs_mask)
        input_weights_a = F.softmax(input_weights_e,dim=-1)
        # (batch_size, encoder_output_dim)
        attended_input = weighted_sum(encoder_outputs, input_weights_a)
        #H*_t = sum(h_i*at_i)
        # (batch_size, encoder_output_dim + decoder_hidden_dim)
        return input_weights_e,input_weights_a,torch.cat((decoder_hidden_state,attended_input), -1)
        # [S_t,H*_t]

    @staticmethod
    def _get_loss(logits: torch.LongTensor,
                  targets: torch.LongTensor,
                  target_mask: torch.LongTensor) -> torch.LongTensor:
        """
        Takes logits (unnormalized outputs from the decoder) of size (batch_size,
        num_decoding_steps, num_classes), target indices of size (batch_size, num_decoding_steps+1)
        and corresponding masks of size (batch_size, num_decoding_steps+1) steps and computes cross
        entropy loss while taking the mask into account.

        The length of ``targets`` is expected to be greater than that of ``logits`` because the
        decoder does not need to compute the output corresponding to the last timestep of
        ``targets``. This method aligns the inputs appropriately to compute the loss.

        During training, we want the logit corresponding to timestep i to be similar to the target
        token from timestep i + 1. That is, the targets should be shifted by one timestep for
        appropriate comparison.  Consider a single example where the target has 3 words, and
        padding is to 7 tokens.
           The complete sequence would correspond to <S> w1  w2  w3  <E> <P> <P>
           and the mask would be                     1   1   1   1   1   0   0
           and let the logits be                     l1  l2  l3  l4  l5  l6
        We actually need to compare:
           the sequence           w1  w2  w3  <E> <P> <P>
           with masks             1   1   1   1   0   0
           against                l1  l2  l3  l4  l5  l6
           (where the input was)  <S> w1  w2  w3  <E> <P>
        """
        relevant_targets = targets[:, 1:].contiguous()  # (batch_size, num_decoding_steps)
        relevant_mask = target_mask[:, 1:].contiguous()  # (batch_size, num_decoding_steps)
        loss = sequence_cross_entropy_with_logits(logits, relevant_targets, relevant_mask)
        return loss

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        This method overrides ``Model.decode``, which gets called after ``Model.forward``, at test
        time, to finalize predictions. The logic for the decoder part of the encoder-decoder lives
        within the ``forward`` method.

        This method trims the output predictions to the first end symbol, replaces indices with
        corresponding tokens, and adds a field called ``predicted_tokens`` to the ``output_dict``.
        """
        predicted_indices = output_dict["predictions"]
        if not isinstance(predicted_indices, numpy.ndarray):
            predicted_indices = predicted_indices.data.cpu().numpy()
        all_predicted_tokens = []
        for indices in predicted_indices:
            indices = list(indices)
            # Collect indices till the first end_symbol
            if self._end_index in indices:
                indices = indices[:indices.index(self._end_index)]
            predicted_tokens = [self.vocab.get_token_from_index(x, namespace=self._target_namespace)
                                for x in indices]
            all_predicted_tokens.append(predicted_tokens)
        if self._pointer_gen:
            self.vocab.drop_extend()
        output_dict["predicted_tokens"] = all_predicted_tokens
        return output_dict

    @overrides
    def forward_on_instances(self,
                             instances: List[Instance],
                             cuda_device: int) -> List[Dict[str, numpy.ndarray]]:
        """
        Takes a list of  :class:`~allennlp.data.instance.Instance`s, converts that text into
        arrays using this model's :class:`Vocabulary`, passes those arrays through
        :func:`self.forward()` and :func:`self.decode()` (which by default does nothing)
        and returns the result.  Before returning the result, we convert any
        ``torch.autograd.Variables`` or ``torch.Tensors`` into numpy arrays and separate the
        batched output into a list of individual dicts per instance. Note that typically
        this will be faster on a GPU (and conditionally, on a CPU) than repeated calls to
        :func:`forward_on_instance`.
        """
        model_input = {}
        dataset = Batch(instances)
        dataset.index_instances(self.vocab)
        model_input.update({'raw':dataset.as_tensor_dict(cuda_device=cuda_device, for_training=False)})
        #extend
        extend_vocab = Vocabulary.from_instances(dataset.instances)
        self.vocab.extend_from(extend_vocab)
        dataset.index_instances(self.vocab)
        model_input.update({'extended':dataset.as_tensor_dict(cuda_device=cuda_device, for_training=False)})
        #input
        model_input.update({'predict':True})
        outputs = self.decode(self(**model_input))

        instance_separated_output: List[Dict[str, numpy.ndarray]] = [{} for _ in dataset.instances]
        for name, output in list(outputs.items()):
            if isinstance(output, torch.autograd.Variable):
                output = output.data.cpu().numpy()
            outputs[name] = output
            for instance_output, batch_element in zip(instance_separated_output, output):
                instance_output[name] = batch_element
        return instance_separated_output

    @classmethod
    def from_params(cls, vocab, params: Params) -> 'PointerGenerator':
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
        scheduled_sampling_ratio = params.pop_float("scheduled_sampling_ratio", 0.0)
        pointer_gen = params.pop_bool("pointer_gen", True)
        max_oovs = params.pop("max_oovs", None)
        params.assert_empty(cls.__name__)
        return cls(vocab,
                   source_embedder=source_embedder,
                   encoder=encoder,
                   max_decoding_steps=max_decoding_steps,
                   target_namespace=target_namespace,
                   attention_function=attention_function,
                   scheduled_sampling_ratio=scheduled_sampling_ratio,
                   pointer_gen=pointer_gen,
                   max_oovs=max_oovs)
