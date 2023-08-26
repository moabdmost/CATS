import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer

def fake_real(query, model_keyword='best-model_detector'):
   
    # device='cuda' if torch.cuda.is_available() else 'cpu'
    device='cpu'
    data = torch.load('/workspace/models/'+ model_keyword +'.pt', map_location='cpu')

    model_name = 'roberta-base'
    
    model = RobertaForSequenceClassification.from_pretrained(model_name)
    tokenizer = RobertaTokenizer.from_pretrained(model_name)

    model.load_state_dict(data['model_state_dict'])
    model.eval()

    tokens = tokenizer.encode(query, add_special_tokens=True, truncation=True, max_length=tokenizer.model_max_length, padding='max_length')
    tokens = torch.tensor(tokens).unsqueeze(0)
    

    with torch.no_grad():
        logits = model(tokens.to(device))[0]
        probs = logits.softmax(dim=-1)

    fake, real = probs.detach().cpu().flatten().numpy().tolist()
  

    if fake > real:
    # Get the predicted label and its probability score
        real_label = 'Fake'
        like_score = 1-real
    else:
        real_label = 'Real'
        like_score = real       


    print("Predicted label:", real_label)
    print("Probability score:", like_score)

    return real_label, like_score