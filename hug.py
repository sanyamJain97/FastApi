from transformers import pipeline

from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import json 
app = FastAPI()

origins = [
  
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

f = open('laptop.json',)
data = json.load(f)

context_ans = ""
# print("ss",data["qAndA"])

for i in data["qAndA"]:
    context_ans += i["answer"] + " " 
    # print(i["answer"])

print(context_ans)


# print("Hello Wolrd")

qa_pipeline = pipeline(
    "question-answering",
    model="mrm8488/bert-small-finetuned-squadv2",
    tokenizer="mrm8488/bert-small-finetuned-squadv2"
)

# abc = qa_pipeline({
#     'context': "Manuel Romero has been working hardly in the repository hugginface/transformers lately",
#     'question': "Who has been working hard for hugginface/transformers lately?"

# })

# fake_items_db = [{"item_name": "Foo"}, {"item_name": "Bar"}, {"item_name": "Baz"}]

# @app.get("/")
# def read_root():
#     return {"Hello": "World"}


# @app.get("/items/{item_id}")
# def read_item(item_id: str, q: Optional[str] = None):
#     return {"item_id": item_id, "q": q}

# @app.get("/items/")
# async def read_item(skip: int = 0, limit: int = 10):
#     print("sss",skip,limit)
#     return fake_items_db[skip : skip + limit]

@app.get("/question")
def read_item(q: Optional[str] = None):
    # return {"q": q}
    print("question",q)
    abc = qa_pipeline({
    'context': context_ans,
    'question': q
    })
    print(abc)
    return {"ans": abc}



# print(abc)