from transformers import AutoModelforCasualLM ,AutoTokenizer
import torch
from lm_eval import evaluator
class model_initial :
  def __init__(self,name):
   
    self.model = AutoModelforCasualLM.from_pretrained(name)
    self.token = AutoTokenizer.from_pretrained(name)
    self.model.eval()


   
    

class Gradients:

    def __init__(self, obj,input):

        text = input

        self.tokens = obj.token(
            text,
            return_tensors="pt",
            padding=True
        )

        self.input_ids = self.tokens["input_ids"]

        # forward pass
        self.outputs = obj.model(
            input_ids=self.input_ids,
            labels=self.input_ids
        )

        self.loss = self.outputs.loss

        obj.model.zero_grad()

        # backward pass
        self.loss.backward()

        self.gradients = {}

        for name, param in obj.model.named_parameters():

            if param.requires_grad and param.grad is not None:

                self.gradients[name] = param.grad.clone()
    


class fisher_info:
   def __init__(self,gradients,type):
      self.fisher_info={}
      if type == "fine" :
         
       for name ,grad in gradients.items():

           self.fisher_info[name]=grad**2
    
     
      elif type =="coarse":

        for name ,grad in gradients.items():

           self.fisher_info[name]=grad**2
        else :
           return #hybid  
        




class Pruning:

    def __init__(self, fisher_info, obj, prune_percent=0.2):

        for name, param in obj.model.named_parameters():

            if name in fisher_info:

                importance = fisher_info[name].float()

                threshold = torch.quantile(
                    importance.view(-1),
                    prune_percent
                )

                mask = importance > threshold

                param.data *= mask 




class Evaluation:

    def __init__(self, obj):

        save_path = "./tinyllama_pruned"

        obj.model.save_pretrained(save_path)

        obj.token.save_pretrained(save_path)

       

        self.results = evaluator.simple_evaluate(

            model="hf",

            model_args=f"pretrained={save_path}",

            tasks=[
                "boolq",
                "piqa",
                "hellaswag",
                "arc_easy",
                "arc_challenge"
            ],

            num_fewshot=0,

            batch_size=1
        )

       

        for task, result in self.results["results"].items():

            print(task)

            for metric, value in result.items():

                print(f"{metric}: {value}")

            print()



from peft import (
    LoraConfig,
    get_peft_model
)

config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(
    model,
    config
)
