from pydantic import BaseModel, model_validator
from typing import List

class comment_object(BaseModel):
    comment_id: str | None = None
    comment: str

class comments_all(BaseModel):
    comments: List[comment_object]
    contains_id: bool = True

    @model_validator(mode="after")
    def check_id(self):

        if all(x.comment_id is not None for x in self.comments):
            return self
        
        elif all(x.comment_id is None for x in self.comments):
            virtual_id = 100

            for item in self.comments:
                if item.comment_id is None:
                    self.contains_id = False
                    item.comment_id = str(virtual_id)
                    virtual_id+=1
                    
            return self
        
        else:
            error_msg = "error: mixed comment ids"
            raise ValueError(error_msg)
            
        

            