from pydantic import BaseModel, model_validator
from typing import List

class texts(BaseModel):
    text_id: str | None = None
    text: str

class payload(BaseModel):
    texts: List[texts]
    contains_id: bool = True

    @model_validator(mode="after")
    def check_id(self):

        if all(x.text_id is not None for x in self.texts):
            return self
        
        elif all(x.text_id is None for x in self.texts):
            virtual_id = 100

            for item in self.texts:
                if item.text_id is None:
                    self.contains_id = False
                    item.text_id = str(virtual_id)
                    virtual_id+=1
                    
            return self
        
        else:
            error_msg = "error: mixed comment ids"
            raise ValueError(error_msg)
            
        

            