import streamlit as st
import torch
from model import SmallLanguageModel
import office_config
st.image("the_office.png", use_column_width=True)
my_config = office_config

chars = office_config.chars
vocab_size = len(chars)
char_to_id = {ch: id for id, ch in enumerate(chars)}
id_to_char = {id: ch for id, ch in enumerate(chars)}
encode = lambda s: [char_to_id[ch] for ch in s]
decode = lambda l: ''.join([id_to_char[id] for id in l])

model = SmallLanguageModel(my_config)
optimizer = torch.optim.AdamW(model.parameters(), lr=my_config.lr)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load('./saved_models/model_6_overfit.pt', map_location=device)

model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
model_config = checkpoint['config']

st.title("The  Office Script Generator")

user_input = st.text_input("Enter input text:")

if st.button("Generate Text"):
    context = torch.tensor(encode(user_input), dtype=torch.long).unsqueeze(0).to(device)
    num_samples = 3 
    st.write("-"*100)
    for i in range(num_samples):
        generated_text = decode(((model.generate(context, 400, 10,1.2))[0]).tolist())
        st.write("### Generated Text "+str(i+1)+":")
        for line in generated_text.split('\n'):
            with st.container():
                st.write(line)
        st.write("-"*100)
                

