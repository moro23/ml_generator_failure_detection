## import libraries
import requests

generator_sample = {
'V1' : -0.613488805,
'V2' : -3.819639825,
'V3' : 2.202301696,
'V4' : 1.300420227,
'V5' : -1.184929415,
'V6' : -4.495963703,
'V7' : -1.83581722,
'V8' : 4.722989459,
'V9' : 1.206140192,
'V10': -0.341909099,
'V11': -5.122874496,
'V12' : 1.017020596,
'V13': 4.818548566,
'V14': 3.269000834,
'V15': -2.984329797,
'V16': 1.387369902,
'V17': 2.032002439,
'V18' : -0.511586796,
'V19' : -1.023069004,
'V20' : 7.338733104,
'V21' : -2.242244047,
'V22' : 0.155489326,
'V23' : 2.053785963,
'V24' : -2.772272579,
'V25' : 1.851369194,
'V26' : -1.788696457,
'V27' : -0.277281712,
'V28' : -1.255143048,
'V29' : -3.832885893,
'V30' : -1.504542368,
'V31' : 1.586765209,
'V32' : 2.291204462,
'V33' : -5.411388233,
'V34' : 0.870072619,
'V35' : 0.574479318,
'V36' : 4.157190971,
'V37' : 1.428093424,
'V38' : -10.5113424,
'V39' : 0.454664278,
'V40' : -1.44836301,
}

url = 'http://127.0.0.1:9696/predict'
response = requests.post(url, json=generator_sample)
result = response.json()
print(result)