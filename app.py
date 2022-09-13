import time
import streamlit as st
import torch
import string
from sgpt import SGPTModel
from DCPCSE import DCPCSEModel
from SimCSE import SimCSEModel
from io import StringIO 
import pdb




from transformers import BertTokenizer, BertForMaskedLM

model_names = [
            {"name":"SGPT-125M", "model":"Muennighoff/SGPT-125M-weightedmean-nli-bitfit","desc":"Smaller variant of SOTA model in information retrieval","url":"https://arxiv.org/abs/2202.08904v5","mark":True,"class":"SGPTModel"},
            {"name":"SGPT-5.8B","model": "Muennighoff/SGPT-5.8B-weightedmean-msmarco-specb-bitfit" ,"desc":"#1 in Information retrieval on CQADupStack dataset (as     of 6 Sept 2022)","url":"https://arxiv.org/abs/2202.08904v5","mark":False,"class":"SGPTModel"},
            {"name":"DCPCSE","model":"DCPCSE/models/large","desc":"#1 in sentence similarity on SICK dataset (as of 6 Sept 2022)","url":"https://arxiv.org/abs/2203.06875v1","mark":True,"class":"DCPCSEModel"},
            {"name":"SIMCSE" ,"model":"princeton-nlp/sup-simcse-roberta-large","desc":"#2 in sentence similarity on SICK dataset (as of 6 Sept 2022)","url":"https://arxiv.org/abs/2104.08821v4","mark":True,"class":"SimCSEModel"},
            {"name":"SIMCSE" ,"model":"princeton-nlp/sup-simcse-roberta-base","desc":"Smaller variant od #2 SOTA model in sentence similarity on SICK dataset (as of 6 Sept 2022)","url":"https://arxiv.org/abs/2104.08821v4","mark":False,"class":"SimCSEModel"},
            {"name":"SGPT-1.3B","model": "Muennighoff/SGPT-1.3B-weightedmean-msmarco-specb-bitfit","desc":"Variant of SOTA model in information retrieval","url":"https://arxiv.org/abs/2202.08904v5","mark":False,"class":"SGPTModel"},
            ]



example_file_display_names = [
"Machine learning terms",
"Customer feedback mixed with sentences unrelated to customer feedback"
]

example_file_names = {
"Machine learning terms": "tests/small_test.txt",
"Customer feedback mixed with sentences unrelated to customer feedback": "tests/larger_test.txt"
}


def construct_model_info_for_display():
    options_arr  = []
    markdown_str = "<div style=\"font-size:16px; color: #2f2f2f; text-align: left\"><br/><b>Models evaluated</b></div>"
    for node in model_names:
        options_arr .append(node["name"] + " - " +  node["desc"])
        if (node["mark"] == True):
            markdown_str += f"<div style=\"font-size:16px; color: #5f5f5f; text-align: left\">&nbsp;•&nbsp;&nbsp;<a href=\'{node['url']}\' target='_blank'>{node['name']}</a>&nbsp;&nbsp;&nbsp;&nbsp;{node['desc']} </div>"
    return options_arr,markdown_str


st.set_page_config(page_title='TWC - Compare state-of-the-art models for Sentence similarity task', page_icon="logo.jpg", layout='centered', initial_sidebar_state='auto',
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
def compute_similarity(sentences,_model,model_name):
    texts,embeddings = _model.compute_embeddings(sentences,is_file=False)
    results = _model.output_results(None,texts,embeddings)
    return results

def run_test(model_name,sentences,display_area):
    display_area.text("Loading model:" + model_name)
    #load_model.clear()
    model = load_model(model_name)
    display_area.text("Model " + model_name  + " load complete")
    try:
        display_area.text("Computing vectors for sentences")
        results = compute_similarity(sentences,model,model_name)
        display_area.text("Similarity computation complete")
        return results
            
    except Exception as e:
        st.error("Some error occurred during prediction" + str(e))
        st.stop()
    return {}
    

def display_results(main_sentence,results):
    main_sent = f"<div style=\"font-size:16px; color: #2f2f2f; text-align: left\"><b>Main sentence:</b>&nbsp;&nbsp;{main_sentence}</div>"
    body_sent = []
    count = 1
    for key in results:
        body_sent.append(f"<div style=\"font-size:16px; color: #2f2f2f; text-align: left\">{count}]&nbsp;{key}&nbsp;&nbsp;&nbsp;<b>{results[key]:.2f}</b></div>")
        count += 1
    main_sent = main_sent + "\n" + '\n'.join(body_sent)
    st.markdown(main_sent,unsafe_allow_html=True)
 
def main():
  st.markdown("<h4 style='text-align: center;'>Compare state-of-the-art models for sentence similarity task</h4>", unsafe_allow_html=True)


  try:
      
      
      with st.form('twc_form'):

        uploaded_file = st.file_uploader("Step 1. Upload text file or choose example file. The first sentence in file is selected as the main sentence", type=".txt")

        selected_file_index = st.selectbox(label='Example files ',  
                    options = example_file_display_names, index=0,  key = "twc_file")
        st.write("")
        options_arr,markdown_str = construct_model_info_for_display()
        selected_model = st.selectbox(label='Step 2. Select Model',  
                    options = options_arr, index=0,  key = "twc_model")
        st.write("")
        submit_button = st.form_submit_button('Run')

        
        input_status_area = st.empty()
        display_area = st.empty()
        if submit_button:
            start = time.time()
            if uploaded_file is not None:
                sentences = StringIO(uploaded_file.getvalue().decode("utf-8")).read()
            else:
                sentences = open(example_file_names[selected_file_index]).read()
            sentences = sentences.split("\n")[:-1]
            results = run_test(selected_model,sentences,display_area)
            display_area.empty()
            with display_area.container():
                st.text(f"Response time - {time.time() - start:.2f} secs")
                display_results(sentences[0],results)
                #st.json(results)
      
      

  except Exception as e:
    st.error("Some error occurred during loading" + str(e))
    st.stop()  
	
  st.markdown(markdown_str, unsafe_allow_html=True)
  
 

if __name__ == "__main__":
   main()

