import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
from keras.models import load_model
from keras import backend as K
from copy import deepcopy
from pathlib import Path


st.set_option('deprecation.showfileUploaderEncoding', False)
l=["SA",'SP','TA','TP','THK','TEAE','TEPE','RAP','REF','RPP','total']
@st.cache(max_entries=10, ttl=3600,suppress_st_warning=False,allow_output_mutation=True)
def load_models(maps):
    model=load_model((Path.cwd()/'models/categorical_SA_best_weights_c1_loss.h5').__str__(),compile=False)
    model._make_predict_function()
    session = K.get_session()
    model1=load_model((Path.cwd()/'models/categorical_total_best_weights_c1_loss.h5').__str__(),compile=False)
    model1._make_predict_function()
    session1 = K.get_session()
    list_m=[(model,session ),(model1,session1 )]
    print('load model')
#     for i in maps[:-1]:
#         model.load_weights(r'models\categorical_%s_best_weights_c1_loss_weights.h5'%i)
#         #models.append(deepcopy(m))
#         #model=load_model('models/categorical_%s_best_weights_c1_loss_weights.h5'%i,compile=False)
#         model._make_predict_function()
#         model.summary()
#         session = K.get_session()
#         list_m.append((deepcopy(model),session))
    return list_m
    
def predict(image_data, model=None):
    image = ImageOps.fit(image_data, (400,400),Image.ANTIALIAS)
    image = image.convert('RGB')
    image = np.asarray(image)
    #st.image(image, channels='RGB')
    image = (image.astype(np.float32) / 255.0)
    img_reshape = image[np.newaxis,...]
    print(img_reshape.shape)
    if model!= None :
        prediction = model.predict(img_reshape)
        return prediction


def main():
    l=["SA",'SP','TA','TP','THK','TEAE','TEPE','RAP','REF','RPP','total']
    print ('Loading models...\n please wait...')
    list_m=load_models(l)
    print('models load successfully...')

    st.write("""
             # ***Keratoconous detector***
             """
             )
    

    option=[]
    st.write("This is a simple image classification web app to predict KC through SIRIUS maps")
    files = st.file_uploader("Please upload  image(jpg) files", type=["jpg"],accept_multiple_files=True)
    l=["SA",'SP','TA','TP','THK','TEAE','TEPE','RAP','REF','RPP','total']
    for i in files:
        image_data= Image.open(i)
        image = ImageOps.fit(image_data, (100,100),Image.ANTIALIAS)
        image = image.convert('RGB')
        image = np.asarray(image)
        st.image(image, channels='RGB')
        st.write('<style>div.Widget.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
        op=st.selectbox('choose map...',l[:-1],key=i)
        option.append(op )
        l.remove(op)
    l=["SA",'SP','TA','TP','THK','TEAE','TEPE','RAP','REF','RPP','total']
    if st.button('p'):
        p=[]
        for i,j in zip(files,option):        
            imageI = Image.open(i)
            print(list_m)
            K.set_session(list_m[0][1])
            #print(session )
            list_m[0][0].load_weights((Path.cwd()/('models/categorical_%s_best_weights_c1_loss_weights.h5'%(j))).__str__())
            list_m[0][0]._make_predict_function()
            prediction =predict(imageI,list_m[0][0])
            p.extend(prediction .tolist()[0])
        import pandas as pd
        st.write(pd.DataFrame(np.array(p).reshape(-1,3),columns=['KC','NORMAL','SUSPECT'],index=option))
        if len(p)==30:
            p=np.array(p)
            #model1=load_model(r'models\categorical_total_best_weights_c1_loss.h5',compile=False)
            K.set_session(list_m[1][1])
            prediction = list_m[1][0].predict(p[np.newaxis,...])
            st.write(pd.DataFrame(np.array(prediction ).reshape(-1,3),columns=['KC','NORMAL','SUSPECT'],index=[l[-1]]))
            
                
#st.write(Path.cwd())
main()
