import time
import streamlit as st
import torch
import string
from sgpt import SGPTModel
from DCPCSE import DCPCSEModel
from SimCSE import SimCSEModel
from sentence_similarity_hf_model import HFModel
from io import StringIO 
import pdb
import json




from transformers import BertTokenizer, BertForMaskedLM

model_names = [
            {   "name":"SGPT-125M", 
                "model":"Muennighoff/SGPT-125M-weightedmean-nli-bitfit",
                "mark":False,
                "class":"SGPTModel"},


            {   "name":"SGPT-5.8B",
                "model": "Muennighoff/SGPT-5.8B-weightedmean-msmarco-specb-bitfit" ,
                "fork_url":"https://github.com/taskswithcode/sgpt",
                "orig_author_url":"https://github.com/Muennighoff",
                "orig_author":"Niklas Muennighoff",
                "sota_info": {   
                                 "task":"#1 in multiple information retrieval & search tasks",
                                 "sota_link":"https://paperswithcode.com/paper/sgpt-gpt-sentence-embeddings-for-semantic",
                            },
                "paper_url":"https://arxiv.org/abs/2202.08904v5",
                "mark":True,
                "class":"SGPTModel"},

            {   "name":"DCPCSE-large",
                "model":"DCPCSE/models/large",
                "fork_url":"https://github.com/taskswithcode/DCPCSE",
                "orig_author_url":"https://github.com/YJiangcm",
                "orig_author":"Jiang Yuxin 姜宇心",
                "sota_info": {   
                                 "task":"#1 in multiple semantic textual similarity tasks",
                                 "sota_link":"https://paperswithcode.com/paper/deep-continuous-prompt-for-contrastive-1"
                            },
                "paper_url":"https://arxiv.org/abs/2203.06875v1",
                "mark":True,
                "class":"DCPCSEModel","sota_link":"https://paperswithcode.com/sota/semantic-textual-similarity-on-sick"},

            {   "name":"SIMCSE-large" ,
                "model":"princeton-nlp/sup-simcse-roberta-large",
                "fork_url":"https://github.com/taskswithcode/SimCSE",
                "orig_author_url":"https://github.com/princeton-nlp",
                "orig_author":"Princeton Natural Language Processing",
                "sota_info": {   
                                 "task":"Within top 10 in multiple semantic textual similarity tasks",
                                 "sota_link":"https://paperswithcode.com/paper/simcse-simple-contrastive-learning-of"
                            },
                "paper_url":"https://arxiv.org/abs/2104.08821v4",
                "mark":True,
                "class":"SimCSEModel","sota_link":"https://paperswithcode.com/sota/semantic-textual-similarity-on-sick"},

            {   "name":"SIMCSE-base" ,
                "model":"princeton-nlp/sup-simcse-roberta-base",
                "mark":False,
                "class":"SimCSEModel"},

            {   "name":"SGPT-1.3B",
                "model": "Muennighoff/SGPT-1.3B-weightedmean-msmarco-specb-bitfit",
                "mark":False,
                "class":"SGPTModel"},

            {   "name":"sentence-transformers/all-MiniLM-L6-v2", 
                "model":"sentence-transformers/all-MiniLM-L6-v2",
                "fork_url":"https://github.com/taskswithcode/sentence_similarity_hf_model",
                "orig_author_url":"https://github.com/UKPLab",
                "orig_author":"Ubiquitous Knowledge Processing Lab",
                "sota_info": {   
                                 "task":"Nearly 4 million downloads from huggingface",
                                 "sota_link":"https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2"
                            },
                "paper_url":"https://arxiv.org/abs/1908.10084",
                "mark":True,
                "class":"HFModel"},

            ]



example_file_names = {
"Machine learning terms (30+ phrases)": "tests/small_test.txt",
"Customer feedback mixed with noise (50+ sentences)":"tests/larger_test.txt"
}

view_count_file = "view_count.txt"

def get_views():
    ret_val = 0
    if ("view_count" not in st.session_state):
        try:
           data = int(open(view_count_file).read().strip("\n"))
        except:
           data = 0
        data += 1
        ret_val = data
        st.session_state["view_count"] = data
        with open(view_count_file,"w") as fp:
            fp.write(str(data))
    else:
        ret_val = st.session_state["view_count"]
    return "{:,}".format(ret_val)
        

def construct_model_info_for_display():
    options_arr  = []
    markdown_str = "<div style=\"font-size:16px; color: #2f2f2f; text-align: left\"><br/><b>Models evaluated</b></div>"
    for node in model_names:
        options_arr .append(node["name"])
        if (node["mark"] == True):
            markdown_str += f"<div style=\"font-size:16px; color: #5f5f5f; text-align: left\">&nbsp;•&nbsp;Model:&nbsp;<a href=\'{node['paper_url']}\' target='_blank'>{node['name']}</a><br/>&nbsp;&nbsp;&nbsp;&nbsp;Code released by:&nbsp;<a href=\'{node['orig_author_url']}\' target='_blank'>{node['orig_author']}</a><br/>&nbsp;&nbsp;&nbsp;&nbsp;Model info:&nbsp;<a href=\'{node['sota_info']['sota_link']}\' target='_blank'>{node['sota_info']['task']}</a><br/>&nbsp;&nbsp;&nbsp;&nbsp;Forked <a href=\'{node['fork_url']}\' target='_blank'>code</a><br/><br/></div>"
    return options_arr,markdown_str


st.set_page_config(page_title='TWC - Compare state-of-the-art models for Sentence Similarity task', page_icon="logo.jpg", layout='centered', initial_sidebar_state='auto',
            menu_items={
             'Get help': 'http://taskswithcode.com',
             'Report a Bug': "mailto:taskswithcode@gmail.com"})
col,pad = st.columns([85,15])

with col:
    st.image("long_form_logo_with_icon.png")


@st.experimental_memo
def load_model(model_name):
    try:
        ret_model = None
        for node in model_names:
            if (model_name.startswith(node["name"])):
                obj_class = globals()[node["class"]]
                ret_model = obj_class()
                ret_model.init_model(node["model"])
        assert(ret_model is not None)
    except Exception as e:
        st.error("Unable to load model:" + model_name + " " +  str(e))
        pass
    return ret_model

  
@st.experimental_memo
def compute_similarity(sentences,_model,model_name,main_index):
    texts,embeddings = _model.compute_embeddings(sentences,is_file=False)
    results = _model.output_results(None,texts,embeddings,main_index)
    return results

def run_test(model_name,sentences,display_area,main_index):
    display_area.text("Loading model:" + model_name)
    #load_model.clear()
    model = load_model(model_name)
    display_area.text("Model " + model_name  + " load complete")
    try:
        display_area.text("Computing vectors for sentences")
        results = compute_similarity(sentences,model,model_name,main_index)
        display_area.text("Similarity computation complete")
        return results
            
    except Exception as e:
        st.error("Some error occurred during prediction" + str(e))
        st.stop()
    return {}



    

def display_results(orig_sentences,main_index,results):
    main_sent = f"<div style=\"font-size:16px; color: #2f2f2f; text-align: left\"><b>Main sentence:</b>&nbsp;&nbsp;{orig_sentences[main_index]}</div>"
    body_sent = []
    download_data = {}
    for key in results:
        index = orig_sentences.index(key) + 1
        body_sent.append(f"<div style=\"font-size:16px; color: #2f2f2f; text-align: left\">{index}]&nbsp;{key}&nbsp;&nbsp;&nbsp;<b>{results[key]:.2f}</b></div>")
        download_data[key] =  f"{results[key]:.2f}" 
    main_sent = main_sent + "\n" + '\n'.join(body_sent)
    st.markdown(main_sent,unsafe_allow_html=True)
    st.session_state["download_ready"] = json.dumps(download_data,indent=4)


def init_session():
    st.session_state["download_ready"] = None    
    st.session_state["model_name"] = "ss_test"
    st.session_state["main_index"] = 1
    st.session_state["file_name"] = "default"
 
def main():
  init_session()
  st.markdown("<h4 style='text-align: center;'>Compare state-of-the-art models for Sentence Similarity task</h4>", unsafe_allow_html=True)
  st.markdown(f"<div style='color: #9f9f9f; text-align: right'>views:&nbsp;{get_views()}</div>", unsafe_allow_html=True)


  try:
      
      
      with st.form('twc_form'):

        uploaded_file = st.file_uploader("Step 1. Upload text file(one sentence in a line) or choose an example text file below.", type=".txt")

        selected_file_index = st.selectbox(label='Example files ',  
                    options = list(dict.keys(example_file_names)), index=0,  key = "twc_file")
        st.write("")
        options_arr,markdown_str = construct_model_info_for_display()
        selected_model = st.selectbox(label='Step 2. Select Model',  
                    options = options_arr, index=0,  key = "twc_model")
        st.write("")
        main_index = st.number_input('Step 3. Enter index of sentence in file to make it the main sentence:',value=1,min_value = 1)
        st.write("")
        submit_button = st.form_submit_button('Run')

        
        input_status_area = st.empty()
        display_area = st.empty()
        if submit_button:
            start = time.time()
            if uploaded_file is not None:
                st.session_state["file_name"]  = uploaded_file.name
                sentences = StringIO(uploaded_file.getvalue().decode("utf-8")).read()
            else:
                st.session_state["file_name"]  = example_file_names[selected_file_index]
                sentences = open(example_file_names[selected_file_index]).read()
            sentences = sentences.split("\n")[:-1]
            if (len(sentences) < main_index):
                main_index = len(sentences)
                st.info("Selected sentence index is larger than number of sentences in file. Truncating to " + str(main_index)) 
            st.session_state["model_name"] = selected_model
            st.session_state["main_index"] = main_index
            results = run_test(selected_model,sentences,display_area,main_index - 1)
            display_area.empty()
            with display_area.container():
                st.text(f"Response time - {time.time() - start:.2f} secs for {len(sentences)} sentences")
                display_results(sentences,main_index - 1,results)
                #st.json(results)
      st.download_button(
         label="Download results as json",
         data= st.session_state["download_ready"] if st.session_state["download_ready"] != None else "",
         disabled = False if st.session_state["download_ready"] != None else True,
         file_name= (st.session_state["model_name"] + "_" +  str(st.session_state["main_index"]) + "_" + '_'.join(st.session_state["file_name"].split(".")[:-1]) + ".json").replace("/","_"),
         mime='text/json',
         key ="download" 
        )
      
      

  except Exception as e:
    st.error("Some error occurred during loading" + str(e))
    st.stop()  
	
  st.markdown(markdown_str, unsafe_allow_html=True)
  
 

if __name__ == "__main__":
   main()

