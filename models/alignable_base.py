import json
import numpy as np
from typing import List, Optional, Tuple, Union

from models.utils import *
import models.interventions
from models.constants import CONST_QKV_INDICES
        
        
class AlignableModel(nn.Module):
    """
    Generic alignable model. Alignments are specified in the config.
    """
    

    def __init__(
        self, 
        alignable_config,
        model
    ):
        super().__init__()
        # we allow one type intervention per alignment
        self.mode = alignable_config.mode
        intervention_type = alignable_config.alignable_interventions_type
        
        ###
        # We instantiate intervention_layers at locations.
        # Note that the layer name mentioned in the config is
        # abstract. Not the actual module name of the model.
        # 
        # This script will automatically convert abstract
        # name into module name if the model type is supported.
        #
        # To support a new model type, you need to provide a
        # mapping between supported abstract type and module name.
        ###
        self.alignable_representations = {}
        self.interventions = {}
        self._key_collision_counter = {}
        for representation in alignable_config.alignable_representations:
            intervention = intervention_type(
                get_alignable_dimension(model, representation),
                proj_dim=alignable_config.alignable_low_rank_dimension
            )
            if isinstance(
                intervention, 
                models.interventions.TrainbleIntervention
            ):
                intervention = intervention.bfloat16()
            
            alignable_module_hook = get_alignable_module_hook(model, representation)
            
            _key = self._get_representation_key(representation)
            self.alignable_representations[_key] = representation
            self.interventions[_key] = (intervention, alignable_module_hook)
        self.sorted_alignable_keys = sort_alignables_by_topological_order(
            model,
            self.alignable_representations
        )
        
        # model with cache activations
        self.activations = {}
        self.model = model
        self.model_config = model.config
        self.model_type = get_internal_model_type(model)
        self.disable_model_gradients()
        

    def __str__(self):
        """
        Print out basic info about this alignable instance
        """
        attr_dict = {
            "model_type": self.model_type,
            "alignable_interventions_type": self.alignable_interventions_type,
            "alignabls": self.sorted_alignable_keys,
            "mode": self.mode
        }
        return json.dumps(attr_dict, indent=4)


    def _get_representation_key(self, representation):
        """
        Provide unique key for each intervention
        """
        l = representation.alignable_layer
        r = representation.alignable_representation_type
        u = representation.alignable_unit
        n = representation.max_number_of_units
        key_proposal = f"layer.{l}.repr.{r}.unit.{u}.nunit.{n}"
        if key_proposal not in self._key_collision_counter:
            self._key_collision_counter[key_proposal] = 0
        else:
            self._key_collision_counter[key_proposal] += 1
        return f"{key_proposal}#{self._key_collision_counter[key_proposal]}"
    

    def set_temperature(self, temp: torch.Tensor):
        """
        Set temperature if needed
        """
        for k, v in self.interventions.items():
            if isinstance(
                v[0], 
                models.interventions.BoundlessRotatedSpaceIntervention
            ):
                v[0].set_temperature(temp)
    
    
    def disable_model_gradients(self):
        """
        Disable gradient in the model
        """
        # Freeze all model weights
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
            

    def disable_intervention_gradients(self):
        """
        Disable gradient in the trainable intervention
        """
        # Freeze all intervention weights
        pass
        
    
    def set_device(self, device):
        """
        Set device of interventions and the model
        """
        for k, v in self.interventions.items():
            if isinstance(
                v[0], 
                models.interventions.TrainbleIntervention
            ):
                v[0].to(device)
        self.model.to(device)


    def count_parameters(self):
        """
        Set device of interventions and the model
        """
        total_parameters = 0
        for k, v in self.interventions.items():
            if isinstance(
                v[0], 
                models.interventions.TrainbleIntervention
            ):
                total_parameters += count_parameters(v[0])
        return total_parameters       
        
        
    def set_zero_grad(self):
        """
        Set device of interventions and the model
        """
        for k, v in self.interventions.items():
            if isinstance(
                v[0], 
                models.interventions.TrainbleIntervention
            ):
                v[0].zero_grad()

    
    def _gather_intervention_output(
        self, output,
        alignable_representations_key,
        unit_locations
    ) -> torch.Tensor:
        """
        Gather intervening activations from the output based on indices
        """
        original_output = output
        # data structure casting
        if isinstance(output, tuple):
            original_output = output[0]
        # gather subcomponent
        original_output = self._output_to_subcomponent(
            original_output,
            alignable_representations_key
        )
        # gather based on intervention locations
        selected_output = gather_neurons(
            original_output,
            self.alignable_representations[
                alignable_representations_key].alignable_unit,
            unit_locations
        )
        return selected_output


    def _output_to_subcomponent(
        self, output, alignable_representations_key,
    ) -> List[torch.Tensor]:
        """
        Helps to get subcomponent of inputs/outputs of a hook
        
        For instance, we need to separate QKV from a hidden representation
        by slicing the original output
        """
        return output_to_subcomponent_fn_mapping[self.model_type](
            output, 
            self.alignable_representations[
                alignable_representations_key
            ].alignable_representation_type, 
            self.model_config
        )

    
    def _scatter_intervention_output(
        self, output, intervened_representation,
        alignable_representations_key,
        unit_locations
    ) -> torch.Tensor:
        """
        Scatter in the intervened activations in the output
        """
        original_output = output
        # data structure casting
        if isinstance(output, tuple):
            original_output = output[0]

        scatter_intervention_output_fn_mapping[self.model_type](
            original_output, intervened_representation, 
            self.alignable_representations[
                alignable_representations_key
            ].alignable_representation_type,
            unit_locations, self.model_config
        )
        return original_output
    

    def _intervention_getter(
        self, alignable_keys, unit_locations,
    ) -> HandlerList:  
        """
        Create a list of getter handlers that will fetch activations
        """
        handlers = []
        for key_i, key in enumerate(alignable_keys):
            _, alignable_module_hook = self.interventions[key]
            def hook_callback(model, input, output=None):
                selected_output = self._gather_intervention_output(
                    input if output is None else output, key, unit_locations[key_i]
                )
                self.activations[key] = selected_output
            handlers.append(alignable_module_hook(hook_callback))

        return HandlerList(handlers)
    
        
    def _intervention_setter(
        self, alignable_keys, unit_locations_source, 
        unit_locations_base
    ) -> HandlerList: 
        """
        Create a list of setter handlers that will set activations
        """
        handlers = []
        for key_i, key in enumerate(alignable_keys):
            intervention, alignable_module_hook = self.interventions[key]
            def hook_callback(model, input, output=None):
                if output is None:
                    # intervene in the module input with a pre forward hook
                    selected_output = self._gather_intervention_output(
                        input, key, unit_locations_base[key_i]
                    )
                    # intervene with cached activations
                    intervened_representation = do_intervention(
                        selected_output, self.activations[key], intervention)
                    # patched in the intervned activations
                    input = self._scatter_intervention_output(
                        input, intervened_representation,
                        key, unit_locations_base[key_i]
                    )
                else:
                    selected_output = self._gather_intervention_output(
                        output, key, unit_locations_base[key_i]
                    )
                    # intervene with cached activations
                    intervened_representation = do_intervention(
                        selected_output, self.activations[key], intervention)
                    # patched in the intervned activations
                    output = self._scatter_intervention_output(
                        output, intervened_representation,
                        key, unit_locations_base[key_i]
                    )
            handlers.append(alignable_module_hook(hook_callback))
            
        return HandlerList(handlers)
        
    
    def forward(
        self, 
        base, # BATCH * SEQ_LEN
        sources, # NUM_S * BATCH * SEQ_LEN
        unit_locations, # {KEYS: NUM_S * BATCH * NUM_INT_LOC_SOURCE}
    ):
        """
        Main forward function that serves a wrapper to
        actual model forward calls. It will use forward
        hooks to do interventions.

        In essense, sources will lead to getter hooks to
        get activations. We will use these activations to
        intervene on our base example.

        Parameters:
        base: The base example.
        sources: A list of source examples.
        unit_locations_base: The intervention locations of
                             base.
        unit_locations_sources: The intervention locations of
                             sources.
                             
        Return:
        base_output: the non-intervened output of the base
        input.
        counterfactual_outputs: the intervened output of the
        base input.
        """
        assert len(sources) == len(self.sorted_alignable_keys)
        if self.mode == "parallel":
            assert "sources->base" in unit_locations
            unit_locations_sources = unit_locations["sources->base"][0]
            unit_locations_base = unit_locations["sources->base"][1]
            unit_locations_sources = np.array(unit_locations_sources)
            unit_locations_base = np.array(unit_locations_base)
            
            if unit_locations_base.shape != unit_locations_sources.shape:
                assert False, "In parallel mode, base intervention location"\
                              " has to be the same as the sources intervention"\
                              " locations"
        elif self.mode == "serial":
            assert "sources->base" not in unit_locations
            assert len(sources) == len(unit_locations)
            
        batch_size = base["input_ids"].shape[0]
        device = base["input_ids"].device
        # returning un-intervened output without gradients
        with torch.inference_mode():
            base_outputs = self.model(**base)
        
        if self.mode == "parallel":
            
            # for each source, we hook in getters to cache activations
            # at each aligning representations
            for key_i, alignable_key in enumerate(self.sorted_alignable_keys):
                get_handlers = self._intervention_getter(
                    [alignable_key],
                    [torch.tensor(unit_locations_sources[key_i], device=device)],
                )
                _ = self.model(**sources[key_i])
                get_handlers.remove()
            
            # in parallel mode, we swap cached activations all into
            # base at once
            for key_i, alignable_key in enumerate(self.sorted_alignable_keys):
                set_handlers = self._intervention_setter(
                    [alignable_key],
                    [torch.tensor(unit_locations_sources[key_i], device=device)],
                    [torch.tensor(unit_locations_base[key_i], device=device)],
                )
            counterfactual_outputs = self.model(**base)
            set_handlers.remove()
            
        elif self.mode == "serial":
            serialized_set_handlers = HandlerList([])
            for key_i, alignable_key in enumerate(self.sorted_alignable_keys):
                if key_i != len(self.sorted_alignable_keys)-1:
                    unit_locations_key = f"source_{key_i}->source_{key_i+1}"
                else:
                    unit_locations_key = f"source_{key_i}->base"

                unit_locations_source = \
                    unit_locations[unit_locations_key][0][0] # last one as only one intervention
                                                             # per source in serial case
                unit_locations_base = \
                    unit_locations[unit_locations_key][1][0]
                # get activation from source_i
                get_handlers = self._intervention_getter(
                    [alignable_key],
                    [torch.tensor(unit_locations_source, device=device)],
                )
                _ = self.model(**sources[key_i])
                get_handlers.remove()
                # set with intervened activation to source_i+1
                set_handlers = self._intervention_setter(
                    [alignable_key],
                    [torch.tensor(unit_locations_source, device=device)],
                    [torch.tensor(unit_locations_base, device=device)],
                )
                # for setters, we don't remove them.
                serialized_set_handlers.extend(set_handlers)
            counterfactual_outputs = self.model(**base)
            serialized_set_handlers.remove()
                
        return base_outputs, counterfactual_outputs
    
