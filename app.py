
#streamlit 라이브러리를 불러오기
import streamlit as st
#AI모델을 불러오기 위한 joblib 불러오기
import joblib
import pandas as pd

# st를 이용하여 타이틀과 입력 방법을 명시한다.

def user_input_features() :
  subcompany = float(st.sidebar.radio("subcompany: ",('1','2','3')))
  gender = st.sidebar.radio("gender: ",('female','male'))
  age = st.sidebar.number_input("age: ")
  op_time=st.sidebar.number_input("op_time: ")
  area=st.sidebar.radio("area: ",('S','C','Q'))

  data = {'subcompany' : [subcompany],
          'gender' : [gender],
          'age' : [age],
          'op_time' : [op_time],
          'area' : [area],
          }
  data_df = pd.DataFrame(data, index=[0])
  return data_df


st.title('서비스 인터넷 설치 후 불량/양품 여부 예측')
st.markdown('* 우측에 데이터를 입력해주세요')


ohe_station = joblib.load("ohe_station.pkl")
ohe_station2 = joblib.load("ohe_station2.pkl")
ohe_station3 = joblib.load("ohe_station3.pkl")
scaler_call = joblib.load("scaler.pkl")
model_call = joblib.load("LR_model_hw.pkl")


new_x_df = user_input_features()

data_cat1 = ohe_station.transform(new_x_df[['gender']])
data_concat = pd.concat([new_x_df.drop(columns=['gender']),pd.DataFrame(data_cat1, columns=['gender_' + str(col) for col in ohe_station.categories_[0]])], axis=1)

data_cat2 = ohe_station2.transform(data_concat[['subcompany']])
data_concat = pd.concat([data_concat.drop(columns=['subcompany']),pd.DataFrame(data_cat2, columns=['subcompany_' + str(col) for col in ohe_station2.categories_[0]])], axis=1)

data_cat3 = ohe_station3.transform(data_concat[['area']])
data_concat = pd.concat([data_concat.drop(columns=['area']),pd.DataFrame(data_cat3, columns=['area_' + str(col) for col in ohe_station3.categories_[0]])], axis=1)

st.dataframe(data_concat)

data_con_scale = scaler_call.transform(data_concat)
result = model_call.predict(data_con_scale) 

#예측결과를 화면에 뿌려준다. 
st.subheader('결과는 다음과 같습니다.')
st.write('불량, 양품 여부:', result[0])