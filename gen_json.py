import nltk
import os
import sys
import json

# python decode_full_model.py --path=./test_data/ --model_dir=./pretrained/new/  --beam=2 --val
def write_to_json2(article_path, abstracts = None,out_file='test_data/val/0.json'):
    """article and abstract are list of string 
       out_file : output file name 
    """
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    with open(article_path) as f:
        article = f.read()
    article_lines = tokenizer.tokenize(article) 
    
    article = ' '.join(article_lines)
    with open(out_file, 'wb') as writer:

        # Write to tf.Example
        js_example = {}
        js_example['id'] = " "
        js_example['article'] = article_lines
        js_example['abstract'] = " "
        js_serialized = json.dumps(js_example, indent=4).encode()
        writer.write(js_serialized)

if __name__ == "__main__":
    write_to_json2(sys.argv[1])