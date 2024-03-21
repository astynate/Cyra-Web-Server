from flask import Flask, request
from cyra_model.tokenizer import CyraTokenizer
from cyra_model.model import Cyra

app = Flask(__name__)

cyra_tokenizer_path = 'D:/Exider Company/Cyra/trained-models/cyra_tokenizer.pickle'
cyra_tokenizer = CyraTokenizer(cyra_tokenizer_path, 50)

model = Cyra(cyra_tokenizer, 1, 512, 8, 25)

@app.route("/")
def hello_world() -> str:
    context = request.args.get('context')

    if context is None or context == '':
        return '.'

    return model(context)

if __name__ == "__main__":
    app.run(port=8080)