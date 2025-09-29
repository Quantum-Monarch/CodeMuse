import ast
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM
from PySide6.QtWidgets import QLabel,QTextEdit,QVBoxLayout,QWidget
from sentence_transformers import SentenceTransformer
"""style transfer training support model"""
modelembedding = SentenceTransformer("all-MiniLM-L6-v2")  # lightweight & fast

def code_embedding(code_chunk):
    return modelembedding.encode(code_chunk).tolist()

def load_appropriate_model():
    flag=False
    # Check GPU availability and memory
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB

        if gpu_memory >= 16:  # High-end GPU
            modelname="codellama/CodeLlama-7b-Instruct-hf"
            prompta="[INST] Provide detailed,professional comments explaining the purpose,parameters,return values, and any important implementation details of the following python code:"
            promptb="[/INST]"

        elif gpu_memory >= 8:  # Mid-range GPU
            modelname = "codellama/CodeLlama-7b-Instruct-hf"
            flag=True
            prompta="[INST] Provide detailed,professional comments explaining the purpose,parameters,return values, and any important implementation details of the following python code:"
            promptb="[/INST]"
              # Quantized
        else:  # Low-end GPU
            modelname = "Salesforce/codegen-2b"
            prompta="# Write Python docstring for:"
            promptb="#"

    else:  # CPU only
        modelname ="SEBIS/code_trans_t5_large_code_documentation_generation_python_multitask"
        prompta="Write a concise Python docstring for the following function. Do not repeat the code:"
        promptb=""

    return modelname, flag,prompta,promptb
model_name, flag, prompta, promptb = load_appropriate_model()
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

if model_name=="SEBIS/code_trans_t5_large_code_documentation_generation_python_multitask":
    model=AutoModelForSeq2SeqLM.from_pretrained(model_name)
    comment_generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
    tokenizer.padding_side = "right"
else:
    tokenizer.padding_side = "left"
    model=AutoModelForCausalLM.from_pretrained(model_name,device_map="auto",load_in_8bit=flag,torch_dtype=torch.float16 if not flag else None,)
    comment_generator= pipeline("text-generation",model=model,tokenizer=tokenizer)


def get_classes(tree, source):
    classes=[]
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            start_line=node.lineno-1
            end_line=getattr(node, "end_lineno", start_line+10)
            code_lines=source.splitlines()[start_line:end_line]
            chunk_code="\n".join(code_lines)
            classes.append((node.name,chunk_code))

    return classes

def get_functions(tree, source):
    functions=[]
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            start_line=node.lineno-1
            end_line=getattr(node, "end_lineno", start_line+10)
            code_lines=source.splitlines()[start_line:end_line]
            chunk_code="\n".join(code_lines)
            functions.append((node.name,chunk_code))

    return functions
def reviewcomments(scroll_area, commentlist):
    container=QWidget()
    layout=QVBoxLayout(container)

    textedits=[]
    embeddings=[]
    nodenames=[]
    origcomments=[]

    for comment in commentlist:
        #the code snippet
        code_label=QLabel(f"<pre>{comment[0]}</pre>")
        layout.addWidget(code_label)#puts it on the display

        #the comment to edit
        code_comment=QTextEdit(comment[1])
        layout.addWidget(code_comment)
        textedits.append(code_comment)
        #the embeddings and node names, original comment
        origcomments.append(comment[1])
        embeddings.append(comment[3])
        nodenames.append(comment[2])
    scroll_area.setWidget(container)
    scroll_area.setWidgetResizable(True)
    return textedits, origcomments,embeddings, nodenames



def comment_code(file, prompta=prompta, promptb=promptb):
    with open(file, "r") as f:
        source = f.read()
    tree = ast.parse(source)
    commentlist=[]
    chunks=[]
    prompts=[]
    chunks.extend(get_classes(tree,source))
    chunks.extend(get_functions(tree, source))
    for name,chunk in chunks:
        prompt=f"""{prompta}{chunk}{promptb}"""
        prompts.append(prompt)
    start=time.time()
    results= comment_generator(
        prompts,
        max_new_tokens=150,
        do_sample=True,
        batch_size=2,
        pad_token_id=tokenizer.eos_token_id,
    )
    end=time.time()
    print(f"comment generated in {end-start} seconds!")
    for i,result in enumerate(results):
        text=result['generated_text'].strip()
        embedding=code_embedding(chunks[i][1])
        commentlist.append([chunks[i][1],text,chunks[i][0],embedding])

    return commentlist