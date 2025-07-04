from transformers.generation import GenerationMixin
from transformers import AutoTokenizer
import re

import copy
import inspect
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn

from transformers.generation.logits_process import (
    EncoderNoRepeatNGramLogitsProcessor,
    EncoderRepetitionPenaltyLogitsProcessor,
    EpsilonLogitsWarper,
    EtaLogitsWarper,
    ExponentialDecayLengthPenalty,
    ForcedBOSTokenLogitsProcessor,
    ForcedEOSTokenLogitsProcessor,
    ForceTokensLogitsProcessor,
    HammingDiversityLogitsProcessor,
    InfNanRemoveLogitsProcessor,
    LogitNormalization,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    MinNewTokensLengthLogitsProcessor,
    NoBadWordsLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
    PrefixConstrainedLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    SequenceBiasLogitsProcessor,
    SuppressTokensAtBeginLogitsProcessor,
    SuppressTokensLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
    TypicalLogitsWarper,
    UnbatchedClassifierFreeGuidanceLogitsProcessor,
)
from transformers.generation.stopping_criteria import (
    MaxLengthCriteria,
    MaxTimeCriteria,
    StoppingCriteria,
    StoppingCriteriaList,
    validate_stopping_criteria,
)
from transformers.generation import (
    GenerateDecoderOnlyOutput, 
    GenerateEncoderDecoderOutput,
    GenerateBeamDecoderOnlyOutput,
    GenerateBeamEncoderDecoderOutput,
)
if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel
    from .streamers import BaseStreamer

# Typing shortcuts
GenerateNonBeamOutput = Union[GenerateDecoderOnlyOutput, GenerateEncoderDecoderOutput]
GenerateBeamOutput = Union[GenerateBeamDecoderOnlyOutput, GenerateBeamEncoderDecoderOutput]
GenerateOutput = Union[GenerateNonBeamOutput, GenerateBeamOutput]

# Replace with the path to other LLM
tokenizer = AutoTokenizer.from_pretrained("/home/LLMs/CodeLlama-7b-hf/") 

class CustomGenerationMixin(GenerationMixin):    
    def greedy_search(
        self,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        r"""
        Generates sequences of token ids for models with a language modeling head using **greedy decoding** and can be
        used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        <Tip warning={true}>

        In most cases, you do not need to call [`~generation.GenerationMixin.greedy_search`] directly. Use generate()
        instead. For an overview of generation strategies and code examples, check the [following
        guide](../generation_strategies).

        </Tip>


        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.

            max_length (`int`, *optional*, defaults to 20):
                **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
                tokens. The maximum length of the sequence to be generated.
            pad_token_id (`int`, *optional*):
                The id of the *padding* token.
            eos_token_id (`Union[int, List[int]]`, *optional*):
                The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more details.
            output_hidden_states (`bool`, *optional*, defaults to `False`):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more details.
            output_scores (`bool`, *optional*, defaults to `False`):
                Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
            return_dict_in_generate (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            model_kwargs:
                Additional model specific keyword arguments will be forwarded to the `forward` function of the model.
                If model is an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`] or
            `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.

        Examples:

        ```python
        >>> from transformers import (
        ...     AutoTokenizer,
        ...     AutoModelForCausalLM,
        ...     LogitsProcessorList,
        ...     MinLengthLogitsProcessor,
        ...     StoppingCriteriaList,
        ...     MaxLengthCriteria,
        ... )

        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")

        >>> # set pad_token_id to eos_token_id because GPT2 does not have a PAD token
        >>> model.generation_config.pad_token_id = model.generation_config.eos_token_id

        >>> input_prompt = "It might be possible to"
        >>> input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids

        >>> # instantiate logits processors
        >>> logits_processor = LogitsProcessorList(
        ...     [
        ...         MinLengthLogitsProcessor(10, eos_token_id=model.generation_config.eos_token_id),
        ...     ]
        ... )
        >>> stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=20)])

        >>> outputs = model.greedy_search(
        ...     input_ids, logits_processor=logits_processor, stopping_criteria=stopping_criteria
        ... )

        >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ["It might be possible to get a better understanding of the nature of the problem, but it's not"]
        ```"""
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_attentions = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

        this_peer_finished = False  # used by synced_gpus only
        orig_len = input_ids.shape[0]
        while True:
            restrict_lists = [{} for _ in range(orig_len)]
            string = [[] for _ in range(orig_len)]
            type = [[] for _ in range(orig_len)]
            type_index = [[] for _ in range(orig_len)]
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # forward pass to get next token
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need
            
            for idx in range(orig_len):
                next_token_logits = outputs.logits[idx, -1, :].unsqueeze(0)

                end_symbol = 0
                end_dict = {}
                for re_len in range(1, len(input_ids[idx])//2):
                    if torch.equal(input_ids[idx][-re_len:], input_ids[idx][-2 * re_len:-re_len]):
                        end_symbol = input_ids[idx][-re_len].item()
                        restrict_num = 0
                        while torch.equal(input_ids[idx][-re_len:], input_ids[idx][-(2 + restrict_num) * re_len:-(1 + restrict_num) * re_len]):
                            restrict_num += 1
                        end_dict[end_symbol] = restrict_num
                        break
                for i in range(0, len(input_ids[idx])):
                    string[idx].append(tokenizer.convert_ids_to_tokens(input_ids[idx][i:i+1])[0].replace("<0x0A>", "\n"))
                type, type_index = convert_ids_to_type(input_ids, string, type, type_index, idx)
                new_type = split_list_by_newline(type, idx)
                new_type_index = split_list_by_reference(type_index, new_type, idx)
                
                
                if len(new_type) >= 2:
                    repeat_list = find_repeated_prefix(new_type[::-1])
                    re_len = len(repeat_list)
                    if re_len != 0:
                        end_symbol = input_ids[idx][new_type_index[-re_len-1][-1]+1].item()
                        if tuple([item for sublist in new_type[-re_len:] for item in sublist]) in restrict_lists[idx]:
                            restrict_lists[idx][tuple([item for sublist in new_type[-re_len:] for item in sublist])] += 1
                        else:
                            restrict_lists[idx][tuple([item for sublist in new_type[-re_len:] for item in sublist])] = 1
                        restrict_num = restrict_lists[idx][tuple([item for sublist in new_type[-re_len:] for item in sublist])]
                        end_dict[end_symbol] = restrict_num


                # pre-process distribution
                next_tokens_scores = logits_processor(input_ids[idx].unsqueeze(0), next_token_logits)

                # Store scores, attentions and hidden_states when required
                if return_dict_in_generate:
                    if output_scores:
                        scores += (next_tokens_scores,)
                    if output_attentions:
                        decoder_attentions += (
                            (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                        )
                        if self.config.is_encoder_decoder:
                            cross_attentions += (outputs.cross_attentions,)

                    if output_hidden_states:
                        decoder_hidden_states += (
                            (outputs.decoder_hidden_states,)
                            if self.config.is_encoder_decoder
                            else (outputs.hidden_states,)
                        )
                if len(end_dict.keys()) != 0:
                    for end_symbol, restrict_num in end_dict.items():
                        next_tokens_scores[0][end_symbol] = next_tokens_scores[0][end_symbol]*(0.9**restrict_num)
                    if tokenizer.decode(end_symbol) in restrict_string:
                        next_tokens_scores[0][tokenizer.eos_token_id] = next_tokens_scores[0][tokenizer.eos_token_id]/(0.8**restrict_num)
                # argmax
                next_token = torch.argmax(next_tokens_scores)

                next_tokens = next_token

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id_tensor is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                )

                # stop when each sentence is finished
                if unfinished_sequences.max() == 0:
                    this_peer_finished = True

            # stop if we exceed the maximum length
            if stopping_criteria(input_ids, scores):
                this_peer_finished = True

            if this_peer_finished and not synced_gpus:
                break

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GenerateEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return GenerateDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return input_ids




keyword_list = ['continue', 'await', 'as', 'try', 'or', 'else', 'assert', 'True', 'except', 'test', 'raise', 'return', 'break', 'while', 'nonlocal', 'if', 'async', 'pass', 'global', 'finally', 'elif', 'and', 'import', 'not', 'from', 'is', 'lambda', 'yield', 'for', 'with', 'None', 'False', 'class', 'del', 'in', 'def', 'print', 'assert']
restrict_string = ['assert', 'if', 'elif', 'else', 'print', 'def', "#", "try", "while", "import", "from", "for", "class", "#", "'''", '"""', '\n']


def convert_ids_to_type(input_ids, string, type, type_index, idx):
    j = -1
    for i in range(0, len(input_ids[idx])):
        if j >= i:
            continue
        if '"""' in string[idx][i]:
            j = i + 1
            while j <= len(input_ids[idx]) - 1 and (('"""' not in string[idx][j]) and not ('""' in string[idx][j-1] and '"' in string[idx][j]) and not ('"' in string[idx][j-1] and '""' in string[idx][j])):
                j += 1
            if j != len(input_ids[idx]):
                type[idx].append("COMMNET")
                type_index[idx].append(j)
            else:
                type[idx].append("COMMNET")
                type_index[idx].append(j-1)
            continue
        if i < len(input_ids[idx]) - 1 and (('""' in string[idx][i] and '"' in string[idx][i+1]) or ('"' in string[idx][i] and '""' in string[idx][i+1])):
            j = i + 2
            while j <= len(input_ids[idx]) - 1 and (('"""' not in string[idx][j]) and not ('""' in string[idx][j-1] and '"' in string[idx][j]) and not ('"' in string[idx][j-1] and '""' in string[idx][j])):
                j += 1
            if j != len(input_ids[idx]):
                type[idx].append("COMMNET")
                type_index[idx].append(j)
            else:
                type[idx].append("COMMNET")
                type_index[idx].append(j-1)
            continue
        if "'''" in string[idx][i]:
            j = i + 1
            while j <= len(input_ids[idx]) - 1 and "'''" not in string[idx][j]:
                j += 1
            if j != len(input_ids[idx]):
                type[idx].append("COMMNET")
                type_index[idx].append(j)
            else:
                type[idx].append("COMMNET")
                type_index[idx].append(j-1)
            continue
        if "#" in string[idx][i]:
            j = i + 1
            while j <= len(input_ids[idx]) - 1 and "\n" not in string[idx][j]:
                j += 1
            type[idx].append("COMMNET")
            type_index[idx].append(j-1)
            if j != len(input_ids[idx]):
                type[idx].append(string[idx][j])
                type_index[idx].append(j)
            continue
        if "[" in string[idx][i]:
            index_0 = string[idx][i].find("'")
            index_1 = string[idx][i].find('"')
            index_2 = string[idx][i].find("[")
            if ((index_0 == -1 and index_1 == -1) or (index_2 < index_0 and index_0 != -1) or (index_2 < index_1 and index_1 != -1)) and "]" not in string[idx][i]:
                j = i + 1
                while j <= len(input_ids[idx]) - 1 and "]" not in string[idx][j]:
                    j += 1
                if j != len(input_ids[idx]):
                    type[idx].append("LIST")
                    type_index[idx].append(j)
                else:
                    type[idx].append("LIST")
                    type_index[idx].append(j-1)
            elif index_0 != -1 and index_2 > index_0 and "]" not in string[idx][i]:
                j = i + 1
                while j <= len(input_ids[idx]) - 1 and "'" not in string[idx][j]:
                    j += 1
                if j != len(input_ids[idx]):
                    type[idx].append("STRING")
                    type_index[idx].append(j)
                else:
                    type[idx].append("STRING")
                    type_index[idx].append(j-1)
            elif index_1 != -1 and index_2 > index_1 and "]" not in string[idx][i]:
                j = i + 1
                while j <= len(input_ids[idx]) - 1 and '"' not in string[idx][j]:
                    j += 1
                if j != len(input_ids[idx]):
                    type[idx].append("STRING")
                    type_index[idx].append(j)
                else:
                    type[idx].append("STRING")
                    type_index[idx].append(j-1)
            elif "]" in string[idx][i]:
                type[idx].append("LIST")
                type_index[idx].append(i)
            continue

        if "'" in string[idx][i] and "'''" not in string[idx][i]:
            if string[idx][i].count("'") != 2:
                j = i + 1
                while j <= len(input_ids[idx]) - 1 and "'" not in string[idx][j]:
                    j += 1
                if j != len(input_ids[idx]):
                    type[idx].append("STRING")
                    type_index[idx].append(j)
                else:
                    type[idx].append("STRING")
                    type_index[idx].append(j-1)    
            else:
                type[idx].append("STRING")
                type_index[idx].append(i)
            continue
        if '"' in string[idx][i] and '"""' not in string[idx][i]:
            if string[idx][i].count('"') != 2:
                j = i + 1
                while j <= len(input_ids[idx]) - 1 and '"' not in string[idx][j]:
                    j += 1
                if j != len(input_ids[idx]):
                    type[idx].append("STRING")
                    type_index[idx].append(j)
                else:
                    type[idx].append("STRING")
                    type_index[idx].append(j-1)    
            else:
                type[idx].append("STRING")
                type_index[idx].append(i)
            continue

        if not re.match(r'^▁?[a-zA-Z_0-9]+$', string[idx][i]):
            type[idx].append(string[idx][i])
            type_index[idx].append(i)
        elif string[idx][i].replace("▁","") in keyword_list:
            j = i + 1
            while j <= len(input_ids[idx]) - 1 and re.match(r'^[a-zA-Z_0-9]+$', string[idx][j]):
                j += 1
            if j == i + 1:
                type[idx].append(string[idx][i])
                type_index[idx].append(i)
            else:
                type[idx].append("NAME")
                type_index[idx].append(j-1)
            j -= 1

        elif re.match(r'^▁?[a-zA-Z_][a-zA-Z_0-9]*$', string[idx][i]):
            j = i + 1
            while j <= len(input_ids[idx]) - 1 and re.match(r'^[a-zA-Z_0-9]+$', string[idx][j]):
                j += 1
            type[idx].append("NAME")
            type_index[idx].append(j-1)
            j -= 1
            continue

        if re.match(r'^▁?[0-9]+$', string[idx][i]):
            j = i + 1
            while j <= len(input_ids[idx]) - 1 and re.match(r'^[0-9]+$', string[idx][j]):
                j += 1
            type[idx].append("NUMBER")
            type_index[idx].append(j-1)
            j -= 1
            continue
    return type, type_index

def split_list_by_newline(type, idx):
    result = []
    sublist = []
    for item in type[idx]:
        # sublist.append(item)
        if item == "\n":
            result.append(sublist)
            result.append(["\n"])
            sublist = []
        else:
            sublist.append(item)
    if sublist:  
        result.append(sublist)
    return result

def split_list_by_reference(single_list, nested_list, idx):
    result = []
    index = 0
    for sublist in nested_list:
        length = len(sublist)
        result.append(single_list[idx][index:index + length])
        index += length
    return result

def kasai(s, suffix_arr):
    n = len(s)
    lcp = [0] * n
    inv_suff = [0] * n
    for i in range(n):
        inv_suff[suffix_arr[i]] = i
    k = 0
    for i in range(n):
        if inv_suff[i] == n - 1:
            k = 0
            continue
        j = suffix_arr[inv_suff[i] + 1]
        while (i + k < n and j + k < n and s[i + k] == s[j + k]):
            k += 1
        lcp[inv_suff[i]] = k
        if k > 0:
            k -= 1
    return lcp

def build_suffix_array(s):
    n = len(s)
    suffixes = [(s[i:], i) for i in range(n)]
    suffixes.sort()
    return [i for _, i in suffixes]

def find_repeated_prefix(s):
    n = len(s)
    suffix_arr = build_suffix_array(s)
    lcp = kasai(s, suffix_arr)
    for length in range(1, n // 2 + 1):
        if n % length == 0:
            index1 = 0
            index2 = length
            inv_suff = [0] * n
            for i in range(n):
                inv_suff[suffix_arr[i]] = i
            lcp_value = min(lcp[min(inv_suff[index1], inv_suff[index2]):max(inv_suff[index1], inv_suff[index2])])
            if lcp_value >= length:
                return s[:length]
    return []
