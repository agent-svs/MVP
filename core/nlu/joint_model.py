import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import DebertaV2PreTrainedModel, DebertaV2Model
from typing import Optional, Tuple, Dict, Any

class JointDeBERTa(DebertaV2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        
        self.num_intents = config.num_intents
        self.num_slot_labels = config.num_slot_labels
        self.intent_loss_weight = getattr(config, "intent_loss_weight", 0.6)
        
        self.deberta = DebertaV2Model(config)
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Intent classification head (on the [CLS] token)
        self.intent_classifier = nn.Linear(config.hidden_size, self.num_intents)
        
        # Slot classification head (per-token)
        self.slot_classifier = nn.Linear(config.hidden_size, self.num_slot_labels)
        
        # Initialize weights and apply final processing
        self.post_init()
        
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        intent_labels: Optional[torch.Tensor] = None,
        slot_labels: Optional[torch.Tensor] = None,
        user_token_mask: Optional[torch.Tensor] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Forward pass for the Joint NLU Model.
        """
        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        
        sequence_output = outputs.last_hidden_state
        pooled_output = sequence_output[:, 0, :]  # CLS token representation
        
        sequence_output = self.dropout(sequence_output)
        pooled_output = self.dropout(pooled_output)
        
        intent_logits = self.intent_classifier(pooled_output)
        slot_logits = self.slot_classifier(sequence_output)
        
        loss = None
        intent_loss = None
        slot_loss = None
        
        if intent_labels is not None and slot_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            
            # Intent Loss
            intent_loss = loss_fct(intent_logits.view(-1, self.num_intents), intent_labels.view(-1))
            
            # Slot Loss
            # Typically slot_labels are already -100 for padding and non-first subwords
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = slot_logits.view(-1, self.num_slot_labels)[active_loss]
                active_labels = slot_labels.view(-1)[active_loss]
                slot_loss = loss_fct(active_logits, active_labels)
            else:
                slot_loss = loss_fct(slot_logits.view(-1, self.num_slot_labels), slot_labels.view(-1))
                
            loss = self.intent_loss_weight * intent_loss + (1.0 - self.intent_loss_weight) * slot_loss
            
        return {
            "loss": loss,
            "intent_logits": intent_logits,
            "slot_logits": slot_logits,
            "intent_loss": intent_loss,
            "slot_loss": slot_loss
        }
